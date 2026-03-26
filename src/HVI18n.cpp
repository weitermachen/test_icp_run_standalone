// Author: weitermachen
// Time: 2026-03-24

#include "HVI18n.h"

namespace hvi18n {

namespace {

std::string ReplaceAll(
    std::string input,
    const std::vector<std::pair<std::string, std::string>>& replacements) {
    for (const auto& replacement : replacements) {
        const std::string placeholder = "{" + replacement.first + "}";
        size_t pos = 0;
        while ((pos = input.find(placeholder, pos)) != std::string::npos) {
            input.replace(pos, placeholder.size(), replacement.second);
            pos += replacement.second.size();
        }
    }
    return input;
}

}  // namespace

bool IsSupportedLanguage(int language) {
    return language == static_cast<int>(UIPilotLanguage::ZH_CN) ||
        language == static_cast<int>(UIPilotLanguage::EN_US);
}

std::string Translate(const Dictionary& dict, const std::string& key, int language) {
    const auto it = dict.find(key);
    if (it == dict.end()) {
        return key;
    }

    const LocalizedText& text = it->second;
    if (language == static_cast<int>(UIPilotLanguage::ZH_CN) && !text.zh.empty()) {
        return text.zh;
    }
    if (!text.en.empty()) {
        return text.en;
    }
    if (!text.zh.empty()) {
        return text.zh;
    }
    return key;
}

std::string TranslateFormat(
    const Dictionary& dict,
    const std::string& key,
    int language,
    const std::vector<std::pair<std::string, std::string>>& replacements) {
    return ReplaceAll(Translate(dict, key, language), replacements);
}

}  // namespace hvi18n

