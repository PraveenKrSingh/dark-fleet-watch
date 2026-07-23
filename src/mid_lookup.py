"""
MMSI MID (Maritime Identification Digits) to ISO 3166-1 alpha-2 code.
The MID is the first 3 digits of a vessel's MMSI and identifies its flag state.

This table covers the Gulf littoral states plus the flag-of-convenience
registries that dominate global tanker/cargo traffic through Hormuz.
Extend as needed -- full ITU MID list: https://www.itu.int/en/ITU-R/terrestrial/fmd/Pages/mid.aspx
"""

MID_TO_ISO = {
    # Gulf littoral states
    "422": "ir",  # Iran
    "470": "ae",  # UAE
    "471": "ae",  # UAE
    "426": "om",  # Oman
    "445": "om",  # Oman (secondary)
    "427": "sa",  # Saudi Arabia (also 403)
    "403": "sa",
    "436": "sa",
    "447": "qa",  # Qatar
    "408": "iq",  # Iraq
    "425": "kw",  # Kuwait
    "419": "in",  # India
    "406": "in",

    # Common flag-of-convenience / major shipping registries
    "372": "pa",  # Panama
    "373": "pa",
    "374": "pa",
    "636": "lr",  # Liberia
    "537": "mh",  # Marshall Islands
    "352": "pa",  # (overlaps commonly seen in practice; verify against full ITU table)
    "563": "sg",  # Singapore
    "525": "id",  # Indonesia
    "533": "my",  # Malaysia
    "548": "my",  # Malaysia (secondary)
    "567": "th",  # Thailand
    "574": "vn",  # Vietnam
    "477": "hk",  # Hong Kong
    "412": "cn",  # China
    "413": "cn",
    "414": "cn",
    "416": "tw",  # Taiwan
    "431": "jp",  # Japan
    "432": "jp",
    "441": "kr",  # South Korea
    "440": "kr",
    "232": "gb",  # United Kingdom
    "235": "gb",
    "244": "nl",  # Netherlands
    "211": "de",  # Germany
    "218": "de",
    "247": "it",  # Italy
    "225": "fr",  # France
    "228": "fr",
    "256": "mt",  # Malta
    "215": "mt",
    "209": "cy",  # Cyprus
    "212": "cy",
    "271": "tr",  # Turkey
    "338": "us",  # United States
    "366": "us",
    "367": "us",
    "368": "us",
    "369": "us",
}


def mid_to_iso(mmsi) -> str:
    """Return the lowercase ISO alpha-2 code for a vessel's flag state, or 'xx' if unknown."""
    mid = str(mmsi)[:3]
    return MID_TO_ISO.get(mid, "xx")


def flag_label(mmsi, name: str, status: str = "") -> str:
    """Convenience formatter: '<flag-icons span> name · status' for use in HTML/markdown tooltips."""
    iso = mid_to_iso(mmsi)
    suffix = f" · {status}" if status else ""
    return f'<span class="fi fi-{iso}"></span> {name}{suffix}'
