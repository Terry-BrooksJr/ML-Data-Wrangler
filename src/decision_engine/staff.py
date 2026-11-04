"""Learnosity staff by Zendesk user ID: department, title and seniority.

The payload attaches this to each comment a known staff member wrote
(payload.build_ticket_payload), so the model can weigh what someone says by
whether it falls in their department's area (AREAS) and how senior they are,
see who on an internal team last spoke before asking to re-engage it, and
notice when executives or a CSM are already involved.

ROSTER was built from the Zendesk users list (every Learnosity user) crossed
with the IDs that appear in the ticket export, then labelled by Terry Brooks
on 2026-11-03. OTHER_STAFF_IDS are the rest of the Learnosity users: staff,
but with no department recorded, so their comments show as "internal". Only
IDs and names are kept here, never emails.

leadership.ROSTER stays the source for Support routing; everyone on it is
here too, under SUPPORT (test_decision_data checks it).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Departments. Each is a value the payload's `staff.department` can hold.
SUPPORT = "support"
ENGINEERING = "engineering"
PLATFORM_SRE = "platform_sre"
PRODUCT = "product"
CUSTOMER_SUCCESS = "customer_success"
PARTNERSHIPS = "partnerships"
BUSINESS_DEVELOPMENT = "business_development"
SALES_ENGINEERING = "sales_engineering"
EXECUTIVE_LEADERSHIP = "executive_leadership"

# What each department has the final word on. The prompt renders this table
# (prompts.py), so a new department must be added here to be explained there.
AREAS: Dict[str, str] = {
    SUPPORT: "the ticket and all communication with the customer",
    ENGINEERING: "defects, root cause, fixes, releases and deployments",
    PLATFORM_SRE: "platform stability, capacity, outages and infrastructure",
    PRODUCT: "feature requests, the roadmap, intended behaviour, Won't Do",
    CUSTOMER_SUCCESS: "the account relationship, renewals and churn risk",
    PARTNERSHIPS: "partner relationships and integrations with partners",
    BUSINESS_DEVELOPMENT: "commercial deals and new business",
    SALES_ENGINEERING: "technical scoping for deals and pre-sales questions",
    EXECUTIVE_LEADERSHIP: "company-level decisions, contracts and exceptions",
}

# Seniority, most senior first: in the same department, a more senior
# person's decision outranks a more junior one's.
C_SUITE = "c_suite"
VP = "vp"
MANAGER = "manager"
INDIVIDUAL = "individual"
SENIORITY = (C_SUITE, VP, MANAGER, INDIVIDUAL)

# Organisations. Questionmark staff share the Zendesk instance; they are an
# internal party, never Learnosity API Support.
LEARNOSITY = "learnosity"
QUESTIONMARK = "questionmark"


@dataclass(frozen=True)
class Staff:
    user_id: str  # Zendesk user ID, as a comment's `author` holds it
    name: str
    department: str
    title: str
    seniority: str
    former: bool = False  # has left that role; still shown for old comments
    org: str = LEARNOSITY

    def describe(self) -> dict:
        """What the payload shows for this person's comments."""
        return {
            "name": self.name,
            "department": self.department,
            "title": self.title,
            "seniority": self.seniority,
            "former": self.former,
            "org": self.org,
        }


ROSTER: Tuple[Staff, ...] = (
    Staff("387134954297", "Chris Short", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("384720516977", "Anthony Mai", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("372535774197", "Kit Tumulak", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("1085778551", "Rich Shupe", SUPPORT, "VP of Support", VP),
    Staff("12289991012893", "Will Griffiths", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("361165286529", "Allain Dollete", SUPPORT, "Team Lead", MANAGER),
    Staff("384720527257", "Stephen Sayage", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("385812489257", "Michael Fares", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("12289975534365", "JR Brooks", SUPPORT, "Engineer", INDIVIDUAL),
    Staff("361165271809", "Horacio Balseiro", SUPPORT, "Team Lead", MANAGER),
    Staff("9281613249", "Vikram Pai", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff("377734889717", "Kate Hake", PRODUCT, "VP of Product Management", VP),
    Staff("16633656808861", "Patrick Taylor", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff("1902445129693", "Vlad Manu", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff(
        "361029724105",
        "Conor Walsh",
        ENGINEERING,
        "Former Support Engineer/Current Product Engineer",
        INDIVIDUAL,
    ),
    Staff("22685852360221", "Audrey Auyang", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff(
        "366928996378", "Dan Goldstein", CUSTOMER_SUCCESS, "VP of Customer Success", VP
    ),
    Staff("1151228361", "Michael Sharman", PRODUCT, "VP of Product Management", VP),
    Staff(
        "1085785671",
        "Gonzalo Zawadzki",
        PARTNERSHIPS,
        "Director of Technical Partnerships",
        VP,
    ),
    Staff("33366497853597", "Terry Bu", SUPPORT, "Engineer", INDIVIDUAL),
    Staff(
        "5230856847517",
        "Nicola O'Hara",
        CUSTOMER_SUCCESS,
        "Customer Success Manager",
        INDIVIDUAL,
    ),
    Staff(
        "361043564177",
        "Michal Pienczak",
        ENGINEERING,
        "Staff Software Engineer QA",
        MANAGER,
    ),
    Staff("361041676678", "Olivier Mehani", PLATFORM_SRE, "SRE", INDIVIDUAL),
    Staff("1076685222", "Dmitry Kochnev", ENGINEERING, "Software Engineer", INDIVIDUAL),
    Staff("360266677485", "Nina van der Behr", PRODUCT, "VP of Product Management", VP),
    Staff("391786569478", "Monet Slinowsky", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff(
        "375248489818",
        "Lilly Sievers",
        SUPPORT,
        "Former Support Engineer",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "367362660778",
        "Angelique Dubois",
        SUPPORT,
        "Former Support Engineer",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "361043564217",
        "Phil Broadbent",
        ENGINEERING,
        "Staff Software Engineer",
        MANAGER,
    ),
    Staff(
        "28902635562909",
        "Andressa Rodrigues",
        CUSTOMER_SUCCESS,
        "Customer Success Manager",
        INDIVIDUAL,
    ),
    Staff(
        "383311317598",
        "Samantha Redwin",
        CUSTOMER_SUCCESS,
        "Customer Success Manager",
        INDIVIDUAL,
    ),
    Staff(
        "361395273645", "Mutya Aller", PLATFORM_SRE, "Service Desk Manager", INDIVIDUAL
    ),
    Staff("1086515311", "Adam Bextream", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff(
        "17352655439901",
        "Josh Borton",
        CUSTOMER_SUCCESS,
        "Customer Success Manager",
        INDIVIDUAL,
    ),
    Staff("5158147272989", "Anthony Murphy", ENGINEERING, "SVP Technology, Build", VP),
    Staff(
        "1904037994653",
        "Nick Hutton",
        PRODUCT,
        "Former Product Manager",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "366064503018",
        "Rhiannon Eldridge",
        ENGINEERING,
        "Software Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "367552702638",
        "Ross McNerney",
        CUSTOMER_SUCCESS,
        "Former Customer Success Manager",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "366389330558",
        "Chris Perdue",
        CUSTOMER_SUCCESS,
        "Former VP of Customer Success",
        VP,
        former=True,
    ),
    Staff(
        "378435525797",
        "David Scarratt",
        ENGINEERING,
        "Senior Software Engineer",
        INDIVIDUAL,
    ),
    Staff("360796280437", "Jack Vo", ENGINEERING, "Engineering Manager", MANAGER),
    Staff(
        "1298925131",
        "Sebastian Blanco",
        ENGINEERING,
        "SVP of Strategic Technical Projects",
        VP,
    ),
    Staff(
        "364036611758", "Aaron Ji", ENGINEERING, "Senior Software Engineer", INDIVIDUAL
    ),
    Staff("34539766340125", "Camille Regnier", PRODUCT, "Product Manager", INDIVIDUAL),
    Staff(
        "22496509323677",
        "Douglas Duncan",
        SUPPORT,
        "Technical Support Representative",
        INDIVIDUAL,
        org=QUESTIONMARK,
    ),
    Staff(
        "1582217642",
        "Mark Lynch",
        EXECUTIVE_LEADERSHIP,
        "CSO (Chief Strategy Officer) & Co-founder",
        C_SUITE,
    ),
    Staff(
        "1611003979",
        "Kevin Holloway",
        BUSINESS_DEVELOPMENT,
        "VP Business Development",
        VP,
    ),
    Staff(
        "1316097532",
        "David O'Sullivan",
        SALES_ENGINEERING,
        "VP of Business Development Engineering",
        VP,
    ),
    Staff(
        "4785610849949",
        "Justin Taylor",
        EXECUTIVE_LEADERSHIP,
        "Former Chief Information Officer",
        C_SUITE,
        former=True,
    ),
    Staff(
        "378953080017",
        "Roger Marley",
        ENGINEERING,
        "Staff Software Engineer, Team Lead",
        MANAGER,
    ),
    Staff(
        "5655769207325",
        "William Coppins",
        PLATFORM_SRE,
        "Senior Systems Development Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "33367122220189",
        "Stephen Meehan",
        PLATFORM_SRE,
        "Senior CloudOps Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "22633161142429",
        "Christian Röwenstrunk",
        PRODUCT,
        "Product Manager",
        INDIVIDUAL,
        org=QUESTIONMARK,
    ),
    Staff(
        "30022643008669",
        "David Campbell",
        PLATFORM_SRE,
        "Cloud Operations Team Lead",
        MANAGER,
    ),
    Staff(
        "6425732168477",
        "Jason Lacy",
        PARTNERSHIPS,
        "Former Senior VP OF Global Partnerships",
        VP,
        former=True,
    ),
    Staff(
        "1901091000593",
        "Daleen Rabe",
        PRODUCT,
        "Lead Product Designer - UX/UI",
        MANAGER,
    ),
    Staff(
        "361043564037",
        "Kerem Daldal",
        ENGINEERING,
        "Senior Software Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "383168289877",
        "David Johnson",
        SALES_ENGINEERING,
        "Senior Business Development Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "22633106016413",
        "Tom Gallacher",
        PRODUCT,
        "Former Product Manager, Psychometrics",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "23753468177565",
        "Alan McCabe",
        ENGINEERING,
        "SVP Technology, Questionmark",
        VP,
        org=QUESTIONMARK,
    ),
    Staff(
        "7778423448477",
        "Steve Lay",
        PRODUCT,
        "SVP Product Management, Questionmark",
        VP,
        org=QUESTIONMARK,
    ),
    Staff(
        "361043563997",
        "Yu Zhang",
        ENGINEERING,
        "Former QA Test Engineer",
        INDIVIDUAL,
        former=True,
    ),
    Staff(
        "362892595417",
        "Dok Ó Caoimh",
        EXECUTIVE_LEADERSHIP,
        "Chief Revenue Officer",
        C_SUITE,
    ),
    Staff(
        "4928366437149",
        "Derrick Meyer",
        PLATFORM_SRE,
        "Former SVP Infrastructure / System Operations",
        VP,
        former=True,
    ),
    Staff(
        "19727453197597",
        "Anirudh Mathur",
        ENGINEERING,
        "Senior Software Engineer",
        INDIVIDUAL,
    ),
    Staff(
        "1231124561",
        "Jordan Hume",
        PRODUCT,
        "former Product Manager",
        INDIVIDUAL,
        former=True,
    ),
)

# Learnosity users with no department recorded.
OTHER_STAFF_IDS = frozenset(
    (
        "1031352421",
        "1039445491",
        "1328463451",
        "2968050865",
        "2987364169",
        "5504728345",
        "13553363145",
        "18250965529",
        "360379929998",
        "360380429937",
        "360442685057",
        "360471262938",
        "360490667858",
        "360536317205",
        "360866327638",
        "360953361677",
        "360956683958",
        "361006090017",
        "361011248478",
        "361043563937",
        "361043563977",
        "361043564017",
        "361043564097",
        "361043564137",
        "361043564157",
        "361043564197",
        "361043564237",
        "361043564257",
        "361043564317",
        "362299556697",
        "362788052217",
        "363088134437",
        "363225464858",
        "363323616778",
        "363360615777",
        "363360618577",
        "363360627537",
        "363598878917",
        "364791291517",
        "364947611938",
        "365302970118",
        "366064477838",
        "372298817377",
        "374453886537",
        "374498816458",
        "375149941538",
        "375150221778",
        "376681313517",
        "376740294538",
        "378094780958",
        "378504190978",
        "378604831538",
        "378604833738",
        "379034491718",
        "379861514737",
        "379954894038",
        "381886925637",
        "382027926958",
        "382450343737",
        "384776890657",
        "392204810638",
        "1900348566233",
        "1906643742673",
        "1907473261413",
        "4581624200221",
        "4726575516189",
        "4726582248605",
        "4785611669021",
        "5345391982365",
        "5655765028765",
        "5655770239773",
        "6060332238877",
        "6060354272925",
        "6165196194845",
        "6165261267229",
        "6319916050461",
        "6912601024925",
        "7778429738525",
        "12448365081501",
        "15903129978781",
        "16436040079773",
        "16559664887069",
        "16633646294045",
        "17481029935005",
        "17944158666909",
        "19148246340125",
        "22496449763741",
        "22496483860637",
        "22496501814301",
        "22496516402717",
        "22496550519709",
        "22496551530397",
        "22496553825309",
        "22496558652573",
        "22496561720605",
        "22496563776029",
        "22496570411933",
        "22496575402653",
        "22496604380957",
        "22496619795229",
        "22633132587549",
        "25223737582877",
        "25933469021085",
        "25961465096733",
        "26181189094045",
        "26955998259485",
        "29234975706397",
        "29873973964957",
        "30020727976093",
        "30022605338909",
        "33092363535645",
        "33872673067933",
        "34655553575709",
        "34891372498205",
        "35082896186653",
        "35673640247453",
        "36599088008861",
        "37120722402717",
        "37238641180445",
        "37257261149213",
        "37772447530909",
        "37990362951837",
        "38303111056797",
        "38321367755037",
        "38347434992413",
        "38629224131613",
        "38871961010205",
        "39766871579805",
        "39769081596573",
    )
)

_BY_ID = {s.user_id: s for s in ROSTER}


def lookup(user_id) -> Optional[Staff]:
    """The roster entry for user_id, or None."""
    return _BY_ID.get(str(user_id).strip()) if user_id is not None else None


def is_staff(user_id) -> bool:
    """True for any Learnosity user: on the roster or in OTHER_STAFF_IDS."""
    if user_id is None:
        return False
    return lookup(user_id) is not None or str(user_id).strip() in OTHER_STAFF_IDS


def areas_table() -> str:
    """AREAS as the prompt shows it: one aligned row per department."""
    width = max(len(d) for d in AREAS)
    return "\n".join(f"  {d.ljust(width)}   {area}" for d, area in AREAS.items())
