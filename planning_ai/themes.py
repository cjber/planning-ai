from enum import Enum

from pydantic import BaseModel, field_validator


class Theme(str, Enum):
    climate = "Climate change"
    biodiversity = "Biodiversity and green spaces"
    wellbeing = "Wellbeing and social inclusion"
    great_places = "Great places"
    jobs = "Jobs"
    homes = "Homes"
    infrastructure = "Infrastructure"


class ClimatePolicies(str, Enum):
    CC_NZ = "Net zero carbon new buildings"
    CC_WE = "Water efficiency in new developments"
    CC_DC = "Designing for a changing climate"
    CC_FM = "Flooding and integrated water management"
    CC_RE = "Renewable energy projects and infrastructure"
    CC_CE = "Reducing waste and supporting the circular economy"
    CC_CS = "Supporting land-based carbon sequestration"


class BiodiversityPolicies(str, Enum):
    BG_BG = "Biodiversity and geodiversity"
    BG_GI = "Green infrastructure"
    BG_TC = "Improving Tree Canopy Cover and the Tree Population"
    BG_RC = "River corridors"
    BG_PO = "Protecting open spaces"
    BG_EO = "Providing and enhancing open spaces"


class WellbeingPolicies(str, Enum):
    WS_HD = "Creating healthy new developments"
    WS_CF = "Community, sports and leisure facilities"
    WS_MU = "Meanwhile uses during long term redevelopments"
    WS_IO = "Creating inclusive employment and business opportunities through new developments"
    WS_HS = "Pollution, health and safety"


class GreatPlacesPolicies(str, Enum):
    GP_PP = "People and place responsive design"
    GP_LC = "Protection and enhancement of landscape character"
    GP_GB = "Protection and enhancement of the Cambridge Green Belt"
    GP_QD = "Achieving high quality development"
    GP_QP = "Establishing high quality landscape and public realm"
    GP_HA = "Conservation and enhancement of heritage assets"
    GP_CC = "Adapting heritage assets to climate change"
    GP_PH = "Protection of public houses"


class JobsPolicies(str, Enum):
    J_NE = "New employment and development proposals"
    J_RE = "Supporting the rural economy"
    J_AL = "Protecting the best agricultural land"
    J_PB = "Protecting existing business space"
    J_RW = "Enabling remote working"
    J_AW = "Affordable workspace and creative industries"
    J_EP = "Supporting a range of facilities in employment parks"
    J_RC = "Retail and centres"
    J_VA = "Visitor accommodation, attractions and facilities"
    J_FD = "Faculty development and specialist / language schools"


class HomesPolicies(str, Enum):
    H_AH = "Affordable housing"
    H_ES = "Exception sites for affordable housing"
    H_HM = "Housing mix"
    H_HD = "Housing density"
    H_GL = "Garden land and subdivision of existing plots"
    H_SS = "Residential space standards and accessible homes"
    H_SH = "Specialist housing and homes for older people"
    H_CB = "Self and custom build homes"
    H_BR = "Build to rent homes"
    H_MO = "Houses in multiple occupation (HMOs)"
    H_SA = "Student accommodation"
    H_DC = "Dwellings in the countryside"
    H_RM = "Residential moorings"
    H_RC = "Residential caravan sites"
    H_GT = "Gypsy and Traveller and Travelling Showpeople sites"
    H_CH = "Community-led housing"


class InfrastructurePolicies(str, Enum):
    I_ST = "Sustainable transport and connectivity"
    I_EV = "Parking and electric vehicles"
    I_FD = "Freight and delivery consolidation"
    I_SI = "Safeguarding important infrastructure"
    I_AD = "Aviation development"
    I_EI = "Energy infrastructure masterplanning"
    I_ID = "Infrastructure and delivery"
    I_DI = "Digital infrastructure"


THEME_TO_POLICY_GROUP = {
    Theme.climate: ClimatePolicies,
    Theme.biodiversity: BiodiversityPolicies,
    Theme.wellbeing: WellbeingPolicies,
    Theme.great_places: GreatPlacesPolicies,
    Theme.jobs: JobsPolicies,
    Theme.homes: HomesPolicies,
    Theme.infrastructure: InfrastructurePolicies,
}


class PolicyDetail(BaseModel):
    policy: str
    details: list[str]


class PolicySelection(BaseModel):
    theme: Theme
    policies: list[PolicyDetail]

    @field_validator("policies", mode="before")
    @classmethod
    def validate_policies(cls, policies, info):
        """Ensure policies match the selected theme."""
        if not isinstance(policies, list):
            raise ValueError("Policies must be provided as a list.")

        theme = info.data.get("theme")
        if not theme:
            raise ValueError("Theme must be provided before validating policies.")

        allowed_policies = [p.value for p in THEME_TO_POLICY_GROUP[theme]]
        for policy in policies:
            if policy["policy"] not in allowed_policies:
                raise ValueError(
                    f"Policy '{policy['policy']}' is not valid for theme '{theme.value}'."
                )
        return policies
