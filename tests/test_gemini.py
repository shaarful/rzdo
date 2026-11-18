"""Try one article against the real LLM extraction, without saving anything.

Uses the same prompt, schema and model as the worker and the Process with LLM button.
Replace TITLE, ARTICLE and PUBLISHED_AT below, then run:

    pipenv run python tests/test_gemini.py
"""
import json
import os
import sys
import time
from pathlib import Path

TITLE = "Anthrax concerns rise in northern districts At least 53 treated in Rangpur, Dinajpur and Gaibandha after showing symptoms"

# ISO date, such as "2026-10-27". Leave empty if unknown; relative dates like
# "on Monday" then stay null.
PUBLISHED_AT = "2026-10-27"

ARTICLE = """
Anthrax infections linked to meat from sick cattle have been reported across Rangpur, Dinajpur and Gaibandha, with at least 53 people receiving treatment for symptoms over the past month.

Health officials said the meat consumed by those affected came mainly from cattle raised in Gaibandha’s Sundarganj upazila and neighbouring Pirgacha upazila of Rangpur, where anthrax outbreaks were also reported around the same time last year.

Tap here to add The Daily Star as a trusted source google-preferred-source-badge
Of those affected, 16 people in Rangpur and Dinajpur were diagnosed with anthrax and have since recovered.

In Sundarganj alone, 37 people, including 19 women, 14 men and 4 children, had received treatment for symptoms so far.

The latest outbreak there spread after a sick cow was slaughtered in Bachhati village of Sabarnanda union on September 9 and its meat distributed among 100 families.

Over the past two weeks, people with symptoms have been reported from five unions: Sabarnanda, Tarapur, Kanchibari, Sonaray and Ramjiban.

In Rangpur, 13 people were diagnosed with anthrax in Adampara village under Chaola union of Pirgachha upazila.

Rangpur Civil Surgeon Shaheen Sultana said the cases were detected on September 22.

“Thirteen people, including four members of the same family, were admitted to the Pirgachha Upazila Health Complex, received treatment for ten days, and have since returned home healthy,” she said.

An icddr,b team collected samples from the 13 patients and confirmed anthrax after testing, she added.

Abdul Hakim, a member of Chaola Union Parishad, said the cow had been bought from Sundarganj upazila in neighbouring Gaibandha and slaughtered.

Dinajpur District Livestock Officer Abdur Rahim said no new case had been detected there so far.

Kamrul Hasan, Sundarganj upazila Health and Family Planning Officer, said anthrax primarily spreads to humans through contact with infected livestock, although secondary transmission can occur through contact with an infected person’s blood or pus.

“Anthrax is completely curable if the patients take medicines properly,” he said.

Kamrul said symptoms were first reported in Sundarganj in early September, when six residents of Kanchibari union developed fever and skin blisters after consuming meat from a sick cow.

In Bachhati village, Goleza Begum, 29, said blisters developed on her hands shortly after she cooked and ate meat from the sick cow slaughtered on September 9.

The Sundarganj Upazila Livestock Office collected two meat samples from Bachhati village on September 21 and sent them to the Central Disease Investigation Laboratory.

Both samples tested positive for anthrax in a report received on September 23, said Sundarganj Upazila Livestock Officer KM Iftekharul Islam.

Following the outbreak, six officials from the Institute of Epidemiology, Disease Control and Research and two researchers from icddr,b arrived in Sundarganj from Dhaka on September 24, Kamrul said.

The IEDCR officials stressed mass awareness and veterinary clearance before sick animals are slaughtered in affected areas.

Google News LinkFor all latest news, follow The Daily Star's Google News channel.
Sundarganj UNO Iffat Jahan Tuli said awareness campaigns were being conducted through loudspeaker announcements, administrative meetings and mosque imams.

Officials were also trying to locate people with symptoms who had not sought hospital treatment, she added.

According to the Sundarganj upazila livestock office, the upazila has around 2.54 lakh cattle.

Following a localised outbreak in October 2025, the government began a vaccination programme across 15 unions in July this year, administering 19,000 cattle vaccines so far.

Official guidelines require 100 percent livestock vaccination within a 3km radius of an affected zone.

KM Iftekharul Islam said 11,000 vaccine doses remained in stock, while a requisition had been submitted for another 15,000.

However, icddr,b and IEDCR researchers at the upazila livestock office said anthrax could also spread through goats, sheep and buffaloes.

On October 5 last year, Rozina Begum, 45, wife of Abul Hossain of Pashchim Belka village in Sundarganj, died while undergoing treatment for anthrax symptoms at Rangpur Medical College Hospital.

At the time, upazila health officials and her family told local journalists that she developed blisters on her fingers after cooking and eating meat from a sick goat.

Asked why other livestock were not being vaccinated, Gaibandha District Livestock Oficer Md Zobaidul Kabir said cattle were being prioritised because outbreaks in the region had historically originated from bovine sources, while other livestock were being kept under observation.

"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # regional-language text in a Windows console
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.dev")

    import django

    django.setup()

    from llm.gemini import ArticleError, GeminiExtractor, ServiceError

    extractor = GeminiExtractor()
    started = time.time()
    try:
        result = extractor.extract(TITLE, ARTICLE, PUBLISHED_AT)
    except (ArticleError, ServiceError) as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")
    elapsed = time.time() - started
    extraction = result.extraction

    print(f"\nModel:     {extractor.model}")
    print(f"Time:      {elapsed:.1f} s (limit: 30 s)")
    print(f"Tokens:    {result.prompt_tokens} in, {result.response_tokens} out")
    if result.input_truncated:
        print("Truncated: yes, the article exceeded LLM_MAX_INPUT_CHARS")

    print(f"\nRelevant: {extraction.is_disease_event}")

    print(f"\nSummary, {len(extraction.summary.split())} words:")
    print(extraction.summary)

    print("\nArticle fields:")
    print(json.dumps(
        {
            "reporter_author": extraction.reporter_author,
            "author_affiliation": extraction.author_affiliation,
            "publication_date": extraction.publication_date,
        },
        indent=2,
        ensure_ascii=False,
    ))

    print(f"\nEvents, {len(extraction.events)}:")
    print(json.dumps(
        [event.model_dump() for event in extraction.events], indent=2, ensure_ascii=False,
    ))

    if extraction.translated_body:
        print("\nTranslation:")
        print(extraction.translated_title or "")
        print(extraction.translated_body)
    else:
        print("\nTranslation: none, the article is already in English")


if __name__ == "__main__":
    main()
