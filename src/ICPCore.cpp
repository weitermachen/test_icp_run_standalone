#include "ICPCore.h"

#include <json.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

namespace {

    struct ScaleLevel {
        double voxel_size = 0.0;
        double max_correspondence_distance = 1.0;
        int max_iterations = 50;
    };

    pcl::PointCloud<pcl::PointXYZ>::Ptr RemoveNaNPoints(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input) {
        auto output = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*input, *output, indices);
        return output;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr DownsampleIfNeeded(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
        double voxel_size) {
        if (!input) {
            return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        }
        if (voxel_size <= 0.0) {
            return input;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        auto output = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        voxel.setInputCloud(input);
        voxel.setLeafSize(static_cast<float>(voxel_size),
            static_cast<float>(voxel_size),
            static_cast<float>(voxel_size));
        voxel.filter(*output);
        return output;
    }

    double GetNormalRadius(const ScaleLevel& level)
    {
        
        if (level.voxel_size > 0.0) {
            return std::max(level.voxel_size * 3.0, 1e-3);
        }
        return std::max(level.max_correspondence_distance * 0.25, 1e-3);
    }

    pcl::PointCloud<pcl::Normal>::Ptr EstimateNormals(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        double radius)
    {
        auto normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        if (!cloud || cloud->empty()) {
            return normals;
        }

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);

        auto tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(radius);
        ne.compute(*normals);

        return normals;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr MakePointNormalCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& xyz,
        const pcl::PointCloud<pcl::Normal>::Ptr& normals)
    {
        auto out = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>);
        if (!xyz || !normals || xyz->size() != normals->size()) {
            return out;
        }

        out->resize(xyz->size());
        for (size_t i = 0; i < xyz->size(); ++i) {
            pcl::PointNormal pn;
            pn.x = (*xyz)[i].x;
            pn.y = (*xyz)[i].y;
            pn.z = (*xyz)[i].z;
            pn.normal_x = (*normals)[i].normal_x;
            pn.normal_y = (*normals)[i].normal_y;
            pn.normal_z = (*normals)[i].normal_z;
            (*out)[i] = pn;
        }
        return out;
    }

    bool ShouldSkipGICPAtThisScale(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_level,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_level)
    {
        if (!source_level || !target_level) {
            return true;
        }
        
        if (source_level->size() < 30 || target_level->size() < 30) {
            return true;
        }
        return false;
    }

    bool ShouldSkipLPICPAtThisScale(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_level,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_level)
    {
        if (!source_level || !target_level) {
            return true;
        }
        if (source_level->size() < 30 || target_level->size() < 30) {
            return true;
        }
        return false;
    }


    std::vector<ScaleLevel> BuildScaleSchedule(const ICPParams& params) {
        std::vector<ScaleLevel> levels;

        const double base_voxel =
            (params.voxel_size > 0.0)
            ? params.voxel_size
            : std::max(params.max_correspondence_distance * 0.08, 1e-6);

        ScaleLevel coarse;
        coarse.voxel_size = base_voxel * 2.0;
        coarse.max_correspondence_distance = params.max_correspondence_distance * 2.0;
        coarse.max_iterations = std::max(20, params.max_iterations / 3);

        ScaleLevel middle;
        middle.voxel_size = base_voxel * 1.0;
        middle.max_correspondence_distance = params.max_correspondence_distance * 1.2;
        middle.max_iterations = std::max(30, params.max_iterations / 2);

        ScaleLevel fine;
        fine.voxel_size = (params.voxel_size > 0.0) ? params.voxel_size : 0.0;
        fine.max_correspondence_distance = params.max_correspondence_distance;
        fine.max_iterations = params.max_iterations;

        levels.push_back(coarse);
        levels.push_back(middle);
        levels.push_back(fine);
        return levels;
    }

    bool RunSingleScaleRegistration(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_full,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_full,
        int method,
        const ScaleLevel& level,
        const Eigen::Matrix4f& initial_guess,
        Eigen::Matrix4f& final_transform,
        double& fitness_score,
        std::string& error_message,
        bool force_icp = false) {

        auto source_level = DownsampleIfNeeded(source_full, level.voxel_size);
        auto target_level = DownsampleIfNeeded(target_full, level.voxel_size);

        if (!source_level || !target_level || source_level->empty() || target_level->empty()) {
            error_message = "Source or target point cloud is empty after multi-scale preprocessing.";
            return false;
        }
        if (source_level->size() < 10 || target_level->size() < 10) {
            error_message = "Source or target point cloud has too few points at current scale.";
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ> aligned_dummy;

        //const bool use_gicp = (method == 1 && !force_icp);
        const bool use_gicp = (method == 1 && !force_icp);
        const bool use_lp_icp = (method == 2 && !force_icp);

        if (use_gicp) {
            
            if (ShouldSkipGICPAtThisScale(source_level, target_level)) {
                error_message = "Skip GICP at current scale because point count is too small.";
                return false;
            }

            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            gicp.setMaximumIterations(level.max_iterations);
            gicp.setMaxCorrespondenceDistance(level.max_correspondence_distance);

            
            int k = 10;
            k = std::min<int>(k, static_cast<int>(source_level->size()) - 1);
            k = std::min<int>(k, static_cast<int>(target_level->size()) - 1);
            k = std::max(k, 4);
            gicp.setCorrespondenceRandomness(k);

            gicp.setInputSource(target_level);
            gicp.setInputTarget(source_level);
            gicp.align(aligned_dummy, initial_guess);

            if (!gicp.hasConverged()) {
                error_message = "GICP did not converge at current scale.";
                return false;
            }

            final_transform = gicp.getFinalTransformation();
            fitness_score = gicp.getFitnessScore();
            return true;
        }

        else if (use_lp_icp) {
            if (ShouldSkipLPICPAtThisScale(source_level, target_level)) {
                error_message = "Skip LP-ICP at current scale because point count is too small.";
                return false;
            }

            const double normal_radius = GetNormalRadius(level);

            auto source_normals = EstimateNormals(source_level, normal_radius);
            auto target_normals = EstimateNormals(target_level, normal_radius);

            if (!source_normals || !target_normals ||
                source_normals->size() != source_level->size() ||
                target_normals->size() != target_level->size()) {
                error_message = "Failed to estimate normals for LP-ICP.";
                return false;
            }

            auto source_with_normals = MakePointNormalCloud(source_level, source_normals);
            auto target_with_normals = MakePointNormalCloud(target_level, target_normals);

            if (!source_with_normals || !target_with_normals ||
                source_with_normals->empty() || target_with_normals->empty()) {
                error_message = "Failed to build point-normal clouds for LP-ICP.";
                return false;
            }

            pcl::PointCloud<pcl::PointNormal> aligned_dummy_normals;

            pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> lp_icp;
            lp_icp.setMaximumIterations(level.max_iterations);
            lp_icp.setMaxCorrespondenceDistance(level.max_correspondence_distance);


            lp_icp.setInputSource(target_with_normals);
            lp_icp.setInputTarget(source_with_normals);
            lp_icp.align(aligned_dummy_normals, initial_guess);

            if (!lp_icp.hasConverged()) {
                error_message = "LP-ICP did not converge at current scale.";
                return false;
            }

            final_transform = lp_icp.getFinalTransformation();
            fitness_score = lp_icp.getFitnessScore();
            return true;
        }

        else {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setMaximumIterations(level.max_iterations);
            icp.setMaxCorrespondenceDistance(level.max_correspondence_distance);
            icp.setInputSource(target_level);
            icp.setInputTarget(source_level);
            icp.align(aligned_dummy, initial_guess);

            if (!icp.hasConverged()) {
                error_message = "ICP did not converge at current scale.";
                return false;
            }

            final_transform = icp.getFinalTransformation();
            fitness_score = icp.getFitnessScore();
            return true;
        }
    }

} // namespace

bool ParseTransformJson(const std::string& json_str, Eigen::Matrix4f& transform, std::string& error_message) {
    transform = Eigen::Matrix4f::Identity();
    error_message.clear();

    if (json_str.empty()) {
        return true;
    }

    try {
        const auto j = nlohmann::json::parse(json_str);

        nlohmann::json matrix_json;
        if (j.is_object() && j.contains("matrix")) {
            matrix_json = j.at("matrix");
        }
        else if (j.is_array()) {
            matrix_json = j;
        }
        else {
            error_message = "Initial transform json must be a 4x4 array or an object containing field 'matrix'.";
            return false;
        }

        if (!matrix_json.is_array() || matrix_json.size() != 4) {
            error_message = "Initial transform matrix must have 4 rows.";
            return false;
        }

        for (int r = 0; r < 4; ++r) {
            if (!matrix_json[r].is_array() || matrix_json[r].size() != 4) {
                error_message = "Each row of the initial transform matrix must contain 4 values.";
                return false;
            }
            for (int c = 0; c < 4; ++c) {
                transform(r, c) = matrix_json[r][c].get<float>();
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        error_message = std::string("Failed to parse initial transform json: ") + e.what();
        return false;
    }
}

ICPResult RunPairRegistration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const ICPParams& params) {
    ICPResult result;
    result.aligned_target.reset(new pcl::PointCloud<pcl::PointXYZ>);
    result.merged_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (!source || !target) {
        result.error_message = "Source or target point cloud is null.";
        return result;
    }

    
    auto source_full = RemoveNaNPoints(source);
    auto target_full = RemoveNaNPoints(target);

    if (!source_full || !target_full || source_full->empty() || target_full->empty()) {
        result.error_message = "Source or target point cloud is empty after preprocessing.";
        return result;
    }

    Eigen::Matrix4f current_transform = Eigen::Matrix4f::Identity();
    if (params.use_initial_guess) {
        current_transform = params.initial_guess;
    }

    const auto levels = BuildScaleSchedule(params);

    double last_fitness = -1.0;
    for (size_t i = 0; i < levels.size(); ++i) {
        Eigen::Matrix4f next_transform = current_transform;
        std::string stage_error;


        const bool force_icp = (i == 0);

        const bool ok = RunSingleScaleRegistration(
            source_full,
            target_full,
            params.method,
            levels[i],
            current_transform,
            next_transform,
            last_fitness,
            stage_error,
            force_icp);

        if (!ok) {
            
            //if (params.method == 1 &&
            //    stage_error.find("Skip GICP") != std::string::npos) {
            //    continue;
            //}
            if ((params.method == 1 && stage_error.find("Skip GICP") != std::string::npos) ||
                (params.method == 2 && stage_error.find("Skip LP-ICP") != std::string::npos)) {
                continue;
            }

            result.error_message =
                "Registration failed at level " + std::to_string(i) + ": " + stage_error;
            return result;
        }

        current_transform = next_transform;
    }

    pcl::transformPointCloud(*target_full, *result.aligned_target, current_transform);

    
    *result.merged_cloud = *source_full;
    *result.merged_cloud += *result.aligned_target;

    result.converged = true;
    result.success = true;
    result.transform = current_transform;
    result.fitness_score = last_fitness;
    if (last_fitness >= 0.0) {
        result.rmse = std::sqrt(last_fitness);
    }

    return result;
}