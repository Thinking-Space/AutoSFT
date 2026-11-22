# Synthesis target catalog (fixed; do not invent your own)

target_id may only be chosen from the list below. Each id corresponds to one constraint combination that actually occurs in the evaluation data source — the deduplicated set of each question's `instruction_id_list` in `assets/benchmarks/<benchmark>/input_data.jsonl`.

**id rule**: `<benchmark>.<subclass1>.<subclass2>...`, where each subclass is the part of the constraint name after the colon, sorted alphabetically, with `-` replaced by `_`. Example: `count:keywords_multiple` → `ifbench.keywords_multiple`; `change_case:english_lowercase` + `punctuation:no_comma` → `ifeval.english_lowercase.no_comma`.

Each row: `id` | full constraint name(s) | number of questions hit in the evaluation | sample keys (first 5). A question belongs to exactly one id — its full constraint combination; a combination id tests the ability to satisfy all of its constraints at once.

The instruction id rule above applies to IFEval and IFBench. Code benchmarks use the explicit problem ids listed below and are an optional synthesis extension.

## ifeval (25 single constraints + 156 combinations = 181)

### Single constraints (25)

| id | constraint | questions | sample keys |
|---|---|---|---|
| `ifeval.number_highlighted_sections` | `detectable_format:number_highlighted_sections` | 19 | 1307, 1644, 1646, 167, 168 |
| `ifeval.forbidden_words` | `keywords:forbidden_words` | 19 | 1147, 1580, 1629, 1675, 2028 |
| `ifeval.repeat_prompt` | `combination:repeat_prompt` | 18 | 1480, 1518, 1656, 1686, 1906 |
| `ifeval.response_language` | `language:response_language` | 17 | 1108, 1154, 1259, 1477, 2225 |
| `ifeval.end_checker` | `startend:end_checker` | 17 | 1128, 1220, 1446, 1659, 1893 |
| `ifeval.no_comma` | `punctuation:no_comma` | 16 | 1001, 1162, 1187, 1738, 2245 |
| `ifeval.number_sentences` | `length_constraints:number_sentences` | 14 | 1268, 1381, 179, 1837, 2266 |
| `ifeval.quotation` | `startend:quotation` | 14 | 122, 1776, 2010, 2015, 219 |
| `ifeval.english_lowercase` | `change_case:english_lowercase` | 13 | 1019, 1051, 1087, 1361, 1593 |
| `ifeval.postscript` | `detectable_content:postscript` | 13 | 1246, 1367, 1537, 1620, 1759 |
| `ifeval.number_bullet_lists` | `detectable_format:number_bullet_lists` | 13 | 102, 1286, 1634, 1934, 2078 |
| `ifeval.title` | `detectable_format:title` | 13 | 1322, 1551, 1802, 2097, 2229 |
| `ifeval.frequency` | `keywords:frequency` | 13 | 1203, 1393, 1498, 1733, 1857 |
| `ifeval.number_words` | `length_constraints:number_words` | 13 | 1072, 1092, 1217, 1251, 1258 |
| `ifeval.english_capital` | `change_case:english_capital` | 11 | 1566, 1645, 1999, 2247, 2341 |
| `ifeval.letter_frequency` | `keywords:letter_frequency` | 11 | 1130, 201, 2107, 2265, 2350 |
| `ifeval.capital_word_frequency` | `change_case:capital_word_frequency` | 10 | 1314, 1592, 1996, 2820, 2849 |
| `ifeval.number_placeholders` | `detectable_content:number_placeholders` | 10 | 1005, 1372, 1927, 2136, 2164 |
| `ifeval.constrained_response` | `detectable_format:constrained_response` | 10 | 227, 3749, 3750, 3751, 3752 |
| `ifeval.number_paragraphs` | `length_constraints:number_paragraphs` | 10 | 1082, 1236, 1248, 1377, 1858 |
| `ifeval.two_responses` | `combination:two_responses` | 9 | 1098, 1332, 1591, 1746, 2383 |
| `ifeval.json_format` | `detectable_format:json_format` | 8 | 1075, 1094, 1148, 13, 2404 |
| `ifeval.existence` | `keywords:existence` | 8 | 1531, 2485, 2567, 2602, 2662 |
| `ifeval.nth_paragraph_first_word` | `length_constraints:nth_paragraph_first_word` | 8 | 181, 1954, 2215, 2250, 2590 |
| `ifeval.multiple_sections` | `detectable_format:multiple_sections` | 5 | 1131, 1548, 2023, 2925, 357 |

### Combination constraints (156)

| id | constraint | questions | sample keys |
|---|---|---|---|
| `ifeval.repeat_prompt.no_comma` | `combination:repeat_prompt` + `punctuation:no_comma` | 8 | 1546, 1944, 2063, 2337, 2713 |
| `ifeval.json_format.forbidden_words` | `detectable_format:json_format` + `keywords:forbidden_words` | 7 | 1137, 1242, 1691, 2395, 2649 |
| `ifeval.number_highlighted_sections.number_words` | `detectable_format:number_highlighted_sections` + `length_constraints:number_words` | 7 | 1601, 2844, 2909, 3243, 340 |
| `ifeval.repeat_prompt.title` | `combination:repeat_prompt` + `detectable_format:title` | 4 | 1012, 1561, 16, 2305 |
| `ifeval.number_placeholders.no_comma` | `detectable_content:number_placeholders` + `punctuation:no_comma` | 4 | 1512, 3245, 3350, 3619 |
| `ifeval.forbidden_words.number_sentences` | `keywords:forbidden_words` + `length_constraints:number_sentences` | 4 | 1476, 2041, 2143, 3362 |
| `ifeval.english_lowercase.number_placeholders` | `change_case:english_lowercase` + `detectable_content:number_placeholders` | 3 | 218, 2765, 3035 |
| `ifeval.english_lowercase.number_sentences` | `change_case:english_lowercase` + `length_constraints:number_sentences` | 3 | 1535, 1967, 3534 |
| `ifeval.english_lowercase.number_words` | `change_case:english_lowercase` + `length_constraints:number_words` | 3 | 152, 1781, 2391 |
| `ifeval.two_responses.response_language.no_comma` | `combination:two_responses` + `language:response_language` + `punctuation:no_comma` | 3 | 1582, 2596, 3335 |
| `ifeval.two_responses.no_comma` | `combination:two_responses` + `punctuation:no_comma` | 3 | 1107, 2436, 3281 |
| `ifeval.multiple_sections.quotation` | `detectable_format:multiple_sections` + `startend:quotation` | 3 | 1730, 2889, 3367 |
| `ifeval.number_highlighted_sections.frequency` | `detectable_format:number_highlighted_sections` + `keywords:frequency` | 3 | 1713, 2616, 3680 |
| `ifeval.number_words.quotation` | `length_constraints:number_words` + `startend:quotation` | 3 | 1216, 2751, 30 |
| `ifeval.capital_word_frequency.number_sentences` | `change_case:capital_word_frequency` + `length_constraints:number_sentences` | 2 | 1653, 1834 |
| `ifeval.english_lowercase.number_bullet_lists` | `change_case:english_lowercase` + `detectable_format:number_bullet_lists` | 2 | 1287, 2100 |
| `ifeval.english_lowercase.existence` | `change_case:english_lowercase` + `keywords:existence` | 2 | 1516, 3380 |
| `ifeval.english_lowercase.letter_frequency` | `change_case:english_lowercase` + `keywords:letter_frequency` | 2 | 1122, 3608 |
| `ifeval.repeat_prompt.title.no_comma` | `combination:repeat_prompt` + `detectable_format:title` + `punctuation:no_comma` | 2 | 1627, 3718 |
| `ifeval.repeat_prompt.existence` | `combination:repeat_prompt` + `keywords:existence` | 2 | 1139, 1281 |
| `ifeval.two_responses.number_words` | `combination:two_responses` + `length_constraints:number_words` | 2 | 1793, 3287 |
| `ifeval.number_placeholders.response_language` | `detectable_content:number_placeholders` + `language:response_language` | 2 | 3494, 3744 |
| `ifeval.json_format.existence` | `detectable_format:json_format` + `keywords:existence` | 2 | 2591, 3518 |
| `ifeval.multiple_sections.number_sentences` | `detectable_format:multiple_sections` + `length_constraints:number_sentences` | 2 | 2637, 3429 |
| `ifeval.number_bullet_lists.quotation` | `detectable_format:number_bullet_lists` + `startend:quotation` | 2 | 1325, 334 |
| `ifeval.number_highlighted_sections.number_sentences` | `detectable_format:number_highlighted_sections` + `length_constraints:number_sentences` | 2 | 1823, 3089 |
| `ifeval.number_highlighted_sections.no_comma` | `detectable_format:number_highlighted_sections` + `punctuation:no_comma` | 2 | 1897, 2230 |
| `ifeval.number_highlighted_sections.quotation` | `detectable_format:number_highlighted_sections` + `startend:quotation` | 2 | 2497, 3409 |
| `ifeval.title.number_words` | `detectable_format:title` + `length_constraints:number_words` | 2 | 337, 3442 |
| `ifeval.existence.number_paragraphs` | `keywords:existence` + `length_constraints:number_paragraphs` | 2 | 1375, 1845 |
| `ifeval.existence.number_sentences` | `keywords:existence` + `length_constraints:number_sentences` | 2 | 1265, 2139 |
| `ifeval.existence.no_comma` | `keywords:existence` + `punctuation:no_comma` | 2 | 1508, 3376 |
| `ifeval.forbidden_words.frequency` | `keywords:forbidden_words` + `keywords:frequency` | 2 | 127, 2084 |
| `ifeval.forbidden_words.letter_frequency` | `keywords:forbidden_words` + `keywords:letter_frequency` | 2 | 1402, 3326 |
| `ifeval.forbidden_words.number_paragraphs` | `keywords:forbidden_words` + `length_constraints:number_paragraphs` | 2 | 3315, 3565 |
| `ifeval.forbidden_words.quotation` | `keywords:forbidden_words` + `startend:quotation` | 2 | 1466, 3709 |
| `ifeval.frequency.number_words` | `keywords:frequency` + `length_constraints:number_words` | 2 | 3415, 3538 |
| `ifeval.letter_frequency.number_words` | `keywords:letter_frequency` + `length_constraints:number_words` | 2 | 1880, 1964 |
| `ifeval.letter_frequency.quotation` | `keywords:letter_frequency` + `startend:quotation` | 2 | 163, 2605 |
| `ifeval.response_language.no_comma` | `language:response_language` + `punctuation:no_comma` | 2 | 1739, 3669 |
| `ifeval.response_language.quotation` | `language:response_language` + `startend:quotation` | 2 | 2724, 3623 |
| `ifeval.number_sentences.no_comma` | `length_constraints:number_sentences` + `punctuation:no_comma` | 2 | 1418, 3691 |
| `ifeval.number_sentences.quotation` | `length_constraints:number_sentences` + `startend:quotation` | 2 | 2303, 3672 |
| `ifeval.capital_word_frequency.number_bullet_lists.number_paragraphs` | `change_case:capital_word_frequency` + `detectable_format:number_bullet_lists` + `length_constraints:number_paragraphs` | 1 | 3098 |
| `ifeval.capital_word_frequency.title` | `change_case:capital_word_frequency` + `detectable_format:title` | 1 | 2853 |
| `ifeval.capital_word_frequency.frequency` | `change_case:capital_word_frequency` + `keywords:frequency` | 1 | 3188 |
| `ifeval.capital_word_frequency.frequency.no_comma` | `change_case:capital_word_frequency` + `keywords:frequency` + `punctuation:no_comma` | 1 | 2275 |
| `ifeval.capital_word_frequency.number_paragraphs` | `change_case:capital_word_frequency` + `length_constraints:number_paragraphs` | 1 | 2180 |
| `ifeval.capital_word_frequency.number_paragraphs.quotation` | `change_case:capital_word_frequency` + `length_constraints:number_paragraphs` + `startend:quotation` | 1 | 1670 |
| `ifeval.capital_word_frequency.no_comma` | `change_case:capital_word_frequency` + `punctuation:no_comma` | 1 | 331 |
| `ifeval.capital_word_frequency.quotation` | `change_case:capital_word_frequency` + `startend:quotation` | 1 | 1040 |
| `ifeval.english_capital.postscript` | `change_case:english_capital` + `detectable_content:postscript` | 1 | 2908 |
| `ifeval.english_capital.postscript.frequency` | `change_case:english_capital` + `detectable_content:postscript` + `keywords:frequency` | 1 | 1219 |
| `ifeval.english_capital.multiple_sections` | `change_case:english_capital` + `detectable_format:multiple_sections` | 1 | 1021 |
| `ifeval.english_capital.number_highlighted_sections` | `change_case:english_capital` + `detectable_format:number_highlighted_sections` | 1 | 3456 |
| `ifeval.english_capital.title` | `change_case:english_capital` + `detectable_format:title` | 1 | 2422 |
| `ifeval.english_capital.existence` | `change_case:english_capital` + `keywords:existence` | 1 | 1779 |
| `ifeval.english_capital.forbidden_words` | `change_case:english_capital` + `keywords:forbidden_words` | 1 | 1132 |
| `ifeval.english_capital.letter_frequency` | `change_case:english_capital` + `keywords:letter_frequency` | 1 | 1389 |
| `ifeval.english_capital.letter_frequency.number_words` | `change_case:english_capital` + `keywords:letter_frequency` + `length_constraints:number_words` | 1 | 2416 |
| `ifeval.english_capital.number_sentences` | `change_case:english_capital` + `length_constraints:number_sentences` | 1 | 3276 |
| `ifeval.english_capital.number_sentences.no_comma` | `change_case:english_capital` + `length_constraints:number_sentences` + `punctuation:no_comma` | 1 | 2571 |
| `ifeval.english_capital.number_words` | `change_case:english_capital` + `length_constraints:number_words` | 1 | 1813 |
| `ifeval.english_capital.no_comma` | `change_case:english_capital` + `punctuation:no_comma` | 1 | 3204 |
| `ifeval.english_capital.end_checker` | `change_case:english_capital` + `startend:end_checker` | 1 | 2736 |
| `ifeval.english_lowercase.postscript.number_bullet_lists` | `change_case:english_lowercase` + `detectable_content:postscript` + `detectable_format:number_bullet_lists` | 1 | 3069 |
| `ifeval.english_lowercase.postscript.no_comma` | `change_case:english_lowercase` + `detectable_content:postscript` + `punctuation:no_comma` | 1 | 2216 |
| `ifeval.english_lowercase.number_highlighted_sections` | `change_case:english_lowercase` + `detectable_format:number_highlighted_sections` | 1 | 2372 |
| `ifeval.english_lowercase.title.letter_frequency` | `change_case:english_lowercase` + `detectable_format:title` + `keywords:letter_frequency` | 1 | 279 |
| `ifeval.english_lowercase.existence.frequency` | `change_case:english_lowercase` + `keywords:existence` + `keywords:frequency` | 1 | 1571 |
| `ifeval.english_lowercase.forbidden_words` | `change_case:english_lowercase` + `keywords:forbidden_words` | 1 | 1436 |
| `ifeval.english_lowercase.frequency` | `change_case:english_lowercase` + `keywords:frequency` | 1 | 1153 |
| `ifeval.english_lowercase.frequency.number_paragraphs` | `change_case:english_lowercase` + `keywords:frequency` + `length_constraints:number_paragraphs` | 1 | 2943 |
| `ifeval.english_lowercase.frequency.number_words` | `change_case:english_lowercase` + `keywords:frequency` + `length_constraints:number_words` | 1 | 296 |
| `ifeval.english_lowercase.nth_paragraph_first_word` | `change_case:english_lowercase` + `length_constraints:nth_paragraph_first_word` | 1 | 3434 |
| `ifeval.english_lowercase.quotation` | `change_case:english_lowercase` + `startend:quotation` | 1 | 1843 |
| `ifeval.repeat_prompt.title.frequency` | `combination:repeat_prompt` + `detectable_format:title` + `keywords:frequency` | 1 | 3272 |
| `ifeval.repeat_prompt.existence.forbidden_words` | `combination:repeat_prompt` + `keywords:existence` + `keywords:forbidden_words` | 1 | 3371 |
| `ifeval.repeat_prompt.existence.response_language` | `combination:repeat_prompt` + `keywords:existence` + `language:response_language` | 1 | 3305 |
| `ifeval.repeat_prompt.forbidden_words` | `combination:repeat_prompt` + `keywords:forbidden_words` | 1 | 374 |
| `ifeval.repeat_prompt.frequency` | `combination:repeat_prompt` + `keywords:frequency` | 1 | 3369 |
| `ifeval.repeat_prompt.letter_frequency` | `combination:repeat_prompt` + `keywords:letter_frequency` | 1 | 1129 |
| `ifeval.repeat_prompt.number_words` | `combination:repeat_prompt` + `length_constraints:number_words` | 1 | 332 |
| `ifeval.two_responses.multiple_sections` | `combination:two_responses` + `detectable_format:multiple_sections` | 1 | 3324 |
| `ifeval.two_responses.title` | `combination:two_responses` + `detectable_format:title` | 1 | 1180 |
| `ifeval.two_responses.title.existence` | `combination:two_responses` + `detectable_format:title` + `keywords:existence` | 1 | 2683 |
| `ifeval.two_responses.title.no_comma` | `combination:two_responses` + `detectable_format:title` + `punctuation:no_comma` | 1 | 2704 |
| `ifeval.two_responses.existence` | `combination:two_responses` + `keywords:existence` | 1 | 3748 |
| `ifeval.two_responses.forbidden_words` | `combination:two_responses` + `keywords:forbidden_words` | 1 | 2577 |
| `ifeval.two_responses.letter_frequency` | `combination:two_responses` + `keywords:letter_frequency` | 1 | 2912 |
| `ifeval.number_placeholders.postscript` | `detectable_content:number_placeholders` + `detectable_content:postscript` | 1 | 1619 |
| `ifeval.number_placeholders.number_bullet_lists.number_sentences` | `detectable_content:number_placeholders` + `detectable_format:number_bullet_lists` + `length_constraints:number_sentences` | 1 | 1908 |
| `ifeval.number_placeholders.number_bullet_lists.number_words` | `detectable_content:number_placeholders` + `detectable_format:number_bullet_lists` + `length_constraints:number_words` | 1 | 2284 |
| `ifeval.number_placeholders.number_highlighted_sections` | `detectable_content:number_placeholders` + `detectable_format:number_highlighted_sections` | 1 | 2785 |
| `ifeval.number_placeholders.forbidden_words` | `detectable_content:number_placeholders` + `keywords:forbidden_words` | 1 | 2471 |
| `ifeval.number_placeholders.frequency.number_words` | `detectable_content:number_placeholders` + `keywords:frequency` + `length_constraints:number_words` | 1 | 2246 |
| `ifeval.number_placeholders.nth_paragraph_first_word.no_comma` | `detectable_content:number_placeholders` + `length_constraints:nth_paragraph_first_word` + `punctuation:no_comma` | 1 | 2439 |
| `ifeval.number_placeholders.number_sentences` | `detectable_content:number_placeholders` + `length_constraints:number_sentences` | 1 | 2162 |
| `ifeval.postscript.number_bullet_lists` | `detectable_content:postscript` + `detectable_format:number_bullet_lists` | 1 | 1305 |
| `ifeval.postscript.number_bullet_lists.number_words` | `detectable_content:postscript` + `detectable_format:number_bullet_lists` + `length_constraints:number_words` | 1 | 1980 |
| `ifeval.postscript.number_highlighted_sections.title` | `detectable_content:postscript` + `detectable_format:number_highlighted_sections` + `detectable_format:title` | 1 | 3710 |
| `ifeval.postscript.title` | `detectable_content:postscript` + `detectable_format:title` | 1 | 3131 |
| `ifeval.postscript.existence.nth_paragraph_first_word` | `detectable_content:postscript` + `keywords:existence` + `length_constraints:nth_paragraph_first_word` | 1 | 2549 |
| `ifeval.postscript.existence.no_comma` | `detectable_content:postscript` + `keywords:existence` + `punctuation:no_comma` | 1 | 1825 |
| `ifeval.postscript.number_paragraphs` | `detectable_content:postscript` + `length_constraints:number_paragraphs` | 1 | 143 |
| `ifeval.postscript.end_checker` | `detectable_content:postscript` + `startend:end_checker` | 1 | 1705 |
| `ifeval.multiple_sections.number_bullet_lists` | `detectable_format:multiple_sections` + `detectable_format:number_bullet_lists` | 1 | 1481 |
| `ifeval.multiple_sections.end_checker` | `detectable_format:multiple_sections` + `startend:end_checker` | 1 | 1127 |
| `ifeval.number_bullet_lists.forbidden_words` | `detectable_format:number_bullet_lists` + `keywords:forbidden_words` | 1 | 1936 |
| `ifeval.number_bullet_lists.forbidden_words.frequency` | `detectable_format:number_bullet_lists` + `keywords:forbidden_words` + `keywords:frequency` | 1 | 3345 |
| `ifeval.number_bullet_lists.letter_frequency` | `detectable_format:number_bullet_lists` + `keywords:letter_frequency` | 1 | 2653 |
| `ifeval.number_bullet_lists.number_paragraphs` | `detectable_format:number_bullet_lists` + `length_constraints:number_paragraphs` | 1 | 2118 |
| `ifeval.number_bullet_lists.number_sentences` | `detectable_format:number_bullet_lists` + `length_constraints:number_sentences` | 1 | 3041 |
| `ifeval.number_bullet_lists.number_words` | `detectable_format:number_bullet_lists` + `length_constraints:number_words` | 1 | 136 |
| `ifeval.number_bullet_lists.no_comma` | `detectable_format:number_bullet_lists` + `punctuation:no_comma` | 1 | 2380 |
| `ifeval.number_highlighted_sections.existence` | `detectable_format:number_highlighted_sections` + `keywords:existence` | 1 | 1237 |
| `ifeval.number_highlighted_sections.existence.frequency` | `detectable_format:number_highlighted_sections` + `keywords:existence` + `keywords:frequency` | 1 | 1379 |
| `ifeval.number_highlighted_sections.existence.no_comma` | `detectable_format:number_highlighted_sections` + `keywords:existence` + `punctuation:no_comma` | 1 | 1348 |
| `ifeval.number_highlighted_sections.frequency.number_paragraphs` | `detectable_format:number_highlighted_sections` + `keywords:frequency` + `length_constraints:number_paragraphs` | 1 | 1342 |
| `ifeval.number_highlighted_sections.letter_frequency.number_sentences` | `detectable_format:number_highlighted_sections` + `keywords:letter_frequency` + `length_constraints:number_sentences` | 1 | 1174 |
| `ifeval.number_highlighted_sections.number_paragraphs` | `detectable_format:number_highlighted_sections` + `length_constraints:number_paragraphs` | 1 | 3351 |
| `ifeval.number_highlighted_sections.number_sentences.quotation` | `detectable_format:number_highlighted_sections` + `length_constraints:number_sentences` + `startend:quotation` | 1 | 2035 |
| `ifeval.number_highlighted_sections.number_words.no_comma` | `detectable_format:number_highlighted_sections` + `length_constraints:number_words` + `punctuation:no_comma` | 1 | 1000 |
| `ifeval.number_highlighted_sections.no_comma.end_checker` | `detectable_format:number_highlighted_sections` + `punctuation:no_comma` + `startend:end_checker` | 1 | 3724 |
| `ifeval.title.existence.response_language` | `detectable_format:title` + `keywords:existence` + `language:response_language` | 1 | 3311 |
| `ifeval.title.frequency.no_comma` | `detectable_format:title` + `keywords:frequency` + `punctuation:no_comma` | 1 | 1928 |
| `ifeval.title.letter_frequency` | `detectable_format:title` + `keywords:letter_frequency` | 1 | 2887 |
| `ifeval.title.response_language` | `detectable_format:title` + `language:response_language` | 1 | 3502 |
| `ifeval.title.number_sentences` | `detectable_format:title` + `length_constraints:number_sentences` | 1 | 1262 |
| `ifeval.title.number_sentences.number_words` | `detectable_format:title` + `length_constraints:number_sentences` + `length_constraints:number_words` | 1 | 1392 |
| `ifeval.title.number_sentences.no_comma` | `detectable_format:title` + `length_constraints:number_sentences` + `punctuation:no_comma` | 1 | 3256 |
| `ifeval.existence.forbidden_words` | `keywords:existence` + `keywords:forbidden_words` | 1 | 3386 |
| `ifeval.existence.frequency` | `keywords:existence` + `keywords:frequency` | 1 | 3327 |
| `ifeval.existence.frequency.quotation` | `keywords:existence` + `keywords:frequency` + `startend:quotation` | 1 | 142 |
| `ifeval.existence.letter_frequency` | `keywords:existence` + `keywords:letter_frequency` | 1 | 2169 |
| `ifeval.existence.number_words` | `keywords:existence` + `length_constraints:number_words` | 1 | 260 |
| `ifeval.existence.number_words.no_comma` | `keywords:existence` + `length_constraints:number_words` + `punctuation:no_comma` | 1 | 1069 |
| `ifeval.existence.quotation` | `keywords:existence` + `startend:quotation` | 1 | 1658 |
| `ifeval.forbidden_words.number_words.end_checker` | `keywords:forbidden_words` + `length_constraints:number_words` + `startend:end_checker` | 1 | 3114 |
| `ifeval.forbidden_words.no_comma` | `keywords:forbidden_words` + `punctuation:no_comma` | 1 | 3479 |
| `ifeval.frequency.number_paragraphs` | `keywords:frequency` + `length_constraints:number_paragraphs` | 1 | 2386 |
| `ifeval.frequency.no_comma` | `keywords:frequency` + `punctuation:no_comma` | 1 | 2583 |
| `ifeval.frequency.end_checker` | `keywords:frequency` + `startend:end_checker` | 1 | 1233 |
| `ifeval.letter_frequency.response_language.no_comma` | `keywords:letter_frequency` + `language:response_language` + `punctuation:no_comma` | 1 | 3195 |
| `ifeval.letter_frequency.number_paragraphs` | `keywords:letter_frequency` + `length_constraints:number_paragraphs` | 1 | 1883 |
| `ifeval.letter_frequency.number_sentences` | `keywords:letter_frequency` + `length_constraints:number_sentences` | 1 | 2787 |
| `ifeval.letter_frequency.number_words.no_comma` | `keywords:letter_frequency` + `length_constraints:number_words` + `punctuation:no_comma` | 1 | 1300 |
| `ifeval.letter_frequency.no_comma` | `keywords:letter_frequency` + `punctuation:no_comma` | 1 | 2449 |
| `ifeval.response_language.number_paragraphs` | `language:response_language` + `length_constraints:number_paragraphs` | 1 | 3063 |
| `ifeval.nth_paragraph_first_word.quotation` | `length_constraints:nth_paragraph_first_word` + `startend:quotation` | 1 | 1609 |
| `ifeval.number_paragraphs.end_checker` | `length_constraints:number_paragraphs` + `startend:end_checker` | 1 | 1943 |
| `ifeval.number_paragraphs.quotation` | `length_constraints:number_paragraphs` + `startend:quotation` | 1 | 3294 |
| `ifeval.number_sentences.number_words` | `length_constraints:number_sentences` + `length_constraints:number_words` | 1 | 1879 |
| `ifeval.number_sentences.end_checker` | `length_constraints:number_sentences` + `startend:end_checker` | 1 | 286 |
| `ifeval.number_words.no_comma` | `length_constraints:number_words` + `punctuation:no_comma` | 1 | 1643 |
| `ifeval.no_comma.end_checker` | `punctuation:no_comma` + `startend:end_checker` | 1 | 3084 |

## ifbench (58 single constraints + 19 combinations = 77)

### Single constraints (58)

| id | constraint | questions | sample keys |
|---|---|---|---|
| `ifbench.unique_word_count` | `count:unique_word_count` | 8 | 52, 53, 54, 55, 56 |
| `ifbench.line_indent` | `format:line_indent` | 7 | 73, 74, 75, 76, 77 |
| `ifbench.parentheses` | `format:parentheses` | 7 | 106, 107, 108, 109, 110 |
| `ifbench.quotes` | `format:quotes` | 7 | 118, 119, 120, 121, 122 |
| `ifbench.sub_bullets` | `format:sub-bullets` | 7 | 126, 127, 128, 129, 130 |
| `ifbench.stop_words` | `ratio:stop_words` | 7 | 171, 172, 173, 174, 175 |
| `ifbench.palindrome` | `words:palindrome` | 7 | 234, 235, 236, 237, 238 |
| `ifbench.vowel` | `words:vowel` | 7 | 263, 264, 265, 266, 267 |
| `ifbench.punctuation` | `count:punctuation` | 6 | 35, 36, 37, 38, 39 |
| `ifbench.emoji` | `format:emoji` | 6 | 65, 66, 67, 68, 69 |
| `ifbench.sentence_type` | `ratio:sentence_type` | 6 | 160, 161, 162, 163, 164 |
| `ifbench.consonants` | `words:consonants` | 6 | 206, 207, 208, 209, 210 |
| `ifbench.paragraph_last_first` | `words:paragraph_last_first` | 6 | 241, 242, 243, 244, 245 |
| `ifbench.conjunctions` | `count:conjunctions` | 5 | 5, 6, 7, 8, 9 |
| `ifbench.keywords_multiple` | `count:keywords_multiple` | 5 | 0, 1, 2, 3, 4 |
| `ifbench.numbers` | `count:numbers` | 5 | 17, 18, 19, 20, 21 |
| `ifbench.person_names` | `count:person_names` | 5 | 25, 26, 27, 28, 29 |
| `ifbench.pronouns` | `count:pronouns` | 5 | 30, 31, 32, 33, 34 |
| `ifbench.word_count_range` | `count:word_count_range` | 5 | 41, 42, 43, 44, 45 |
| `ifbench.words_japanese` | `count:words_japanese` | 5 | 47, 48, 49, 50, 51 |
| `ifbench.list` | `format:list` | 5 | 82, 83, 84, 85, 86 |
| `ifbench.newline` | `format:newline` | 5 | 90, 91, 92, 93, 94 |
| `ifbench.no_bullets_bullets` | `format:no_bullets_bullets` | 5 | 95, 96, 97, 98, 99 |
| `ifbench.options` | `format:options` | 5 | 100, 101, 102, 103, 104 |
| `ifbench.quote_unquote` | `format:quote_unquote` | 5 | 113, 114, 115, 116, 117 |
| `ifbench.thesis` | `format:thesis` | 5 | 138, 139, 140, 141, 142 |
| `ifbench.overlap` | `ratio:overlap` | 5 | 144, 145, 146, 147, 148 |
| `ifbench.sentence_balance` | `ratio:sentence_balance` | 5 | 153, 154, 155, 156, 157 |
| `ifbench.sentence_words` | `ratio:sentence_words` | 5 | 166, 167, 168, 169, 170 |
| `ifbench.alliteration_increment` | `sentence:alliteration_increment` | 5 | 178, 179, 180, 181, 182 |
| `ifbench.increment` | `sentence:increment` | 5 | 183, 184, 185, 186, 187 |
| `ifbench.keyword` | `sentence:keyword` | 5 | 191, 192, 193, 194, 195 |
| `ifbench.alphabet` | `words:alphabet` | 5 | 200, 201, 202, 203, 204 |
| `ifbench.last_first` | `words:last_first` | 5 | 219, 220, 221, 222, 223 |
| `ifbench.no_consecutive` | `words:no_consecutive` | 5 | 224, 225, 226, 227, 228 |
| `ifbench.odd_even_syllables` | `words:odd_even_syllables` | 5 | 229, 230, 231, 232, 233 |
| `ifbench.prime_lengths` | `words:prime_lengths` | 5 | 247, 248, 249, 250, 251 |
| `ifbench.repeats` | `words:repeats` | 5 | 252, 253, 254, 255, 256 |
| `ifbench.start_verb` | `words:start_verb` | 5 | 257, 258, 259, 260, 261 |
| `ifbench.no_whitespace` | `format:no_whitespace` | 4 | 296, 297, 298, 299 |
| `ifbench.output_template` | `format:output_template` | 4 | 292, 293, 294, 295 |
| `ifbench.title_case` | `format:title_case` | 4 | 288, 289, 290, 291 |
| `ifbench.repeat_span` | `repeat:repeat_span` | 4 | 284, 285, 286, 287 |
| `ifbench.keywords_specific_position` | `words:keywords_specific_position` | 4 | 10, 11, 12, 13 |
| `ifbench.repeat_change` | `repeat:repeat_change` | 3 | 60, 61, 62 |
| `ifbench.words_position` | `words:words_position` | 3 | 14, 15, 16 |
| `ifbench.repeat_simple` | `repeat:repeat_simple` | 2 | 63, 64 |
| `ifbench.character_reverse` | `custom:character_reverse` | 1 | 273 |
| `ifbench.csv_city` | `custom:csv_city` | 1 | 274 |
| `ifbench.csv_quotes` | `custom:csv_quotes` | 1 | 275 |
| `ifbench.csv_special_character` | `custom:csv_special_character` | 1 | 276 |
| `ifbench.date_format_list` | `custom:date_format_list` | 1 | 277 |
| `ifbench.european_capitals_sort` | `custom:european_capitals_sort` | 1 | 278 |
| `ifbench.mcq_count_length` | `custom:mcq_count_length` | 1 | 279 |
| `ifbench.multiples` | `custom:multiples` | 1 | 280 |
| `ifbench.reverse_newline` | `custom:reverse_newline` | 1 | 281 |
| `ifbench.sentence_alphabet` | `custom:sentence_alphabet` | 1 | 282 |
| `ifbench.word_reverse` | `custom:word_reverse` | 1 | 283 |

### Combination constraints (19)

| id | constraint | questions | sample keys |
|---|---|---|---|
| `ifbench.sub_bullets.stop_words` | `format:sub-bullets` + `ratio:stop_words` | 5 | 133, 134, 135, 136, 137 |
| `ifbench.word_count_range.keyword` | `count:word_count_range` + `sentence:keyword` | 4 | 196, 197, 198, 199 |
| `ifbench.overlap.keyword` | `ratio:overlap` + `sentence:keyword` | 4 | 149, 150, 151, 152 |
| `ifbench.consonants.odd_even_syllables` | `words:consonants` + `words:odd_even_syllables` | 4 | 215, 216, 217, 218 |
| `ifbench.pronouns.vowel` | `count:pronouns` + `words:vowel` | 3 | 270, 271, 272 |
| `ifbench.list.no_consecutive` | `format:list` + `words:no_consecutive` | 3 | 87, 88, 89 |
| `ifbench.thesis.increment` | `format:thesis` + `sentence:increment` | 3 | 188, 189, 190 |
| `ifbench.overlap.consonants` | `ratio:overlap` + `words:consonants` | 3 | 212, 213, 214 |
| `ifbench.conjunctions.numbers` | `count:conjunctions` + `count:numbers` | 2 | 22, 23 |
| `ifbench.emoji.keyword` | `format:emoji` + `sentence:keyword` | 2 | 71, 72 |
| `ifbench.line_indent.sentence_words` | `format:line_indent` + `ratio:sentence_words` | 2 | 80, 81 |
| `ifbench.sentence_balance.consonants` | `ratio:sentence_balance` + `words:consonants` | 2 | 158, 159 |
| `ifbench.numbers.unique_word_count` | `count:numbers` + `count:unique_word_count` | 1 | 24 |
| `ifbench.person_names.start_verb` | `count:person_names` + `words:start_verb` | 1 | 262 |
| `ifbench.word_count_range.list` | `count:word_count_range` + `format:list` | 1 | 46 |
| `ifbench.word_count_range.thesis` | `count:word_count_range` + `format:thesis` | 1 | 143 |
| `ifbench.emoji.quotes` | `format:emoji` + `format:quotes` | 1 | 125 |
| `ifbench.newline.options` | `format:newline` + `format:options` | 1 | 105 |
| `ifbench.parentheses.alphabet` | `format:parentheses` + `words:alphabet` | 1 | 205 |

## EvalPlus code targets (optional)

Code targets identify one problem from the checked-in EvalPlus benchmark catalog. They are selected from evaluation failures in the same way as other Synthesis targets, then generalized by the worker into new training tasks. A target must not contain a reference solution, hidden tests, or an answer copied from a benchmark source.

| benchmark | target id format | problem id range |
|---|---|---|
| HumanEval+ | `humaneval_plus.problem_<n>` | `HumanEval/0` through `HumanEval/163` |
| MBPP+ | `mbpp_plus.problem_<n>` | the checked-in public MBPP+ ids listed below |

The MBPP+ catalog currently contains 378 ids:

```text
2-4, 6-9, 11-12, 14, 16-20, 56-59, 61-72, 74-75, 77, 79-80, 82,
84-106, 108-109, 111, 113, 116, 118-120, 123-133, 135, 137-142, 145,
160-162, 165-168, 170-172, 222-224, 226-227, 230, 232-235, 237-240,
242, 244-245, 247, 250-253, 255-257, 259-262, 264-274, 276-287, 290,
292-294, 296-297, 299-301, 305-306, 308-312, 388-392, 394-395, 397-398,
404-406, 409-410, 412-415, 418-422, 424-430, 432-433, 435-437, 439-441,
445-448, 450-451, 453-460, 462-463, 465, 468, 470-479, 554-560, 562-569,
572-573, 576-581, 583, 585-594, 596-600, 602-608, 610-612, 614-616,
618-620, 622-624, 626, 628-633, 635, 637-639, 641, 643-644, 720-726,
728, 730-737, 739-745, 748-755, 757-760, 762-767, 769-773, 775, 777-778,
780-782, 784-788, 790-794, 796-801, 803-809
```

For each code target, `materials/evalplus_failures.jsonl` is required. Copy raw failure records from the local evaluation run; do not rewrite them. Each record must contain `evidence_schema_version` (`"1.0"`), the matching `benchmark`, positive `repeat`, matching `sample_id` and `problem_id`, `syn_target_id`, non-empty `prompt`, `prediction`, `base_result`, `plus_result`, and a matching `failure_class`. A non-empty `opencompass_result_ref` identifies the local evaluation artifact for compatibility with existing exports; public local evaluators may provide the source-neutral `evaluation_result_ref` field instead. The validator checks structure and failure classification, but does not claim to prove artifact provenance.

Allowed failure classes are `plus_only_failure` for `base_result=pass` and `plus_result=fail`, and `base_and_plus_failure` when both results are `fail`. The evidence must describe a plus-suite failure and may not be an infrastructure or missing-result record. Public evaluation artifacts should contain the prompt and model prediction only; keep benchmark tests and reference implementations out of the materials directory.
