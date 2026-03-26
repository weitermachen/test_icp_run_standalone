// Author: weitermachen
// Time: 2026-03-24

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "3d_pilot_public_def.h"

namespace hvi18n {

struct LocalizedText {
    std::string zh;
    std::string en;
};

using Dictionary = std::unordered_map<std::string, LocalizedText>;

bool IsSupportedLanguage(int language);
std::string Translate(const Dictionary& dict, const std::string& key, int language);
std::string TranslateFormat(
    const Dictionary& dict,
    const std::string& key,
    int language,
    const std::vector<std::pair<std::string, std::string>>& replacements);

}  // namespace hvi18n

