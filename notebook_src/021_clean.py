permits["filed"] = pd.to_datetime(permits.submitted_date, errors="coerce")
permits["permit_date"] = pd.to_datetime(permits.issue_date, errors="coerce")
permits["occupancy_date"] = pd.to_datetime(permits.cofo_date, errors="coerce")
for column in ["valuation", "lat", "lon", "square_footage", "height"]:
    permits[column] = pd.to_numeric(permits[column], errors="coerce")
permits["text"] = permits.work_desc.fillna("").str.upper()
permits = permits[permits.filed.notna()].copy()
DATA_DATE = pd.to_datetime(permits.refresh_time, errors="coerce").max().normalize()

PLAN_CHECK_ONLY = r"ZONING ONLY|ZONING REVIEW ONLY|ZONING PLAN CHECK ONLY|PRELIMINARY ZONING|PDPP|HSAP|PC ONLY|PARALLEL DESIGN|SB ?330"

REAL_BUILDING = r"\d+[- ]?STOR(?:Y|IES).{0,60}(?:UNIT|APARTMENT|OFFICE|RETAIL|HOTEL|WAREHOUSE)"
SMALL_STRUCTURE = (
    r"CARPORT|PATIO|CANOPY|CANOPIES|TRASH ENCLOSURE|\bSIGN\b(?! (?:IS|UNDER|TO BE|PERMIT))|FENCE|RETAINING WALL|\bPOOL\b|\bTENT\b"
    r"|\bTANK\b|ANTENNA|MONOPOLE|SHADE STRUCTURE|SHADE SAIL|GAZEBO|TRELLIS|\bSHED\b|\bBARN\b|GUARD (?:SHACK|BOOTH|HOUSE)|\bBOOTH\b"
    r"|KIOSK|MODULAR|PAY STATION|TEMPORARY|TRAILER|PERGOLA|CABANA|BLEACHER|GENERATOR|EQUIPMENT (?:ENCLOSURE|SHELTER|PAD)"
    r"|RESTROOM BUILDING|PUBLIC RESTROOM|\bNEW RESTROOM\b|TOILET ROOM|PREFAB|BUNGALOW|PORTABLE|RELOCATABLE|CABIN|PUMP HOUSE"
    r"|PUMP ROOM|DUGOUT|PLAY ?GROUND|LUNCH SHELTER|COVERED PARKING|BUS SHELTER|BIKE SHELTER|\bVAULT\b|OPEN STAGE"
    r"|(?<!COIN )LAUNDRY ROOM|MAILBOX|COMMERCIAL\s+COACH|\bCOACH\b.{0,20}(?:OFFICE|CLASSROOM)|HCD APPROVED"
    r"|STATE[- ]APPROVED\s+(?:COMMERCIAL\s+|MODULAR\s+)?(?:COACH|BOOTH|STRUCTURE)"
    r"|STORAGE (?:BLDG|BUILDING).{0,30}ACCESSORY TO|ACCESSORY TO.{0,40}STORAGE (?:BLDG|BUILDING)|PRE[- ]?FAB\w* STORAGE"
)
HOUSE = r"\bADU\b|ACCESSORY DWELLING|\bDUPLEX\b|SINGLE[- ]FAMILY (?:DWELLING|HOUSE|RESIDENCE)|\bTWO[- ]UNIT\b|\b2[- ]UNIT\b|ACCESSORY LAUNDRY"
NOT_A_HOUSE = r"\b(?:[3-9]|\d{2,})[- ]?UNITS?\b|THREE[- ]UNIT|\bAPARTMENT\b|MIX|RETAIL|RESTAURANT|COMMERCIAL|OFFICE|TOWNHO"
SMALL_GARAGE = r"\b(?:\d+|TWO|THREE|FOUR)\s*-?\s*CAR\s+GARAGE"
NOT_A_SMALL_GARAGE = r"UNIT|APART|APT|OFFICE|RETAIL|CONDO|TOWNHO|DWELLING|REPAIR|DEALER"
TINY_FOOTPRINT = r"\b(?:1?\d|2[0-5])'[-\d \"/.]*(?:\"|'')?\s*X\s*(?:1?\d|2[0-5])'"
NOT_TINY = r"UNIT|APART|HOTEL|RETAIL|RESTAURANT|\b[2-9]\d?[- ]STOR"


def is_small_structure(text):
    if re.search(REAL_BUILDING, text):
        return False
    if re.search(SMALL_STRUCTURE, text):
        return True
    if re.search(HOUSE, text) and not re.search(NOT_A_HOUSE, text):
        return True
    if re.search(SMALL_GARAGE, text) and not re.search(NOT_A_SMALL_GARAGE, text):
        return True
    return bool(re.search(TINY_FOOTPRINT, text)) and not re.search(NOT_TINY, text)


MIXED_USE = (
    r"MIX(?:ED)?[-/ ]?USED?|LIVE.?WORK"
    r"|(?:APARTMENT|\d+[- ]UNITS? (?:APARTMENT|AFFORDABLE|SENIOR|SUPPORTIVE)|DWELLING UNITS|RESIDENTIAL UNITS).{0,120}(?:RETAIL|COMMERCIAL|RESTAURANT|MEDICAL OFFICE)"
    r"|(?:RETAIL|COMMERCIAL ?(?:SPACE|USE)|RESTAURANT|MEDICAL OFFICE).{0,60}(?:APARTMENT|\d+[- ]UNITS? (?:APARTMENT|RESIDENTIAL|AFFORDABLE))"
    r"|APT/ ?OFFICE|APARTMENT/ ?OFFICE"
)
CARE_HOME = r"ELDER|ASSISTED|CARE FACILITY|TRANSITIONAL LIVING"
NOT_A_HOTEL = r"RESIDENTIAL HOTEL|(?:ADJOINING|ADJACENT TO|ACCESSORY TO|FOR)\s+(?:THE\s+)?(?:EXISTING|\(E\))\s+HOTEL|\bAPARTMENT (?:BUILDING|COMPLEX)\b"
APARTMENTS = r"\d+[- ]UNITS?\s+(?:\w+\s+){0,5}(?:APARTMENT|APT|CONDO)|APARTMENT (?:BUILDING|BLDG|COMPLEX|HOUSE)|\bAPT\b"
CIVIC = (
    r"SCHOOL|CLASSROOM|ACADEMIC|COLLEGE|UNIVERSIT|\bCAMPUS\b|MUSEUM|CHURCH|MOSQUE|SYNAGOGUE|MAUSOLEUM|COLUMBAR|CRYPT|STADIUM|DORM"
    r"|FITNESS CENTER|RECREATION CENTER|COMMUNITY CENTER|PUBLIC LIBRARY|FIRE STATION"
)
PARKING_BUILDING = r"PARKING (?:STRUCTURE|GARAGE|TIER)"
CLEARLY_RETAIL = r"AUTO (?:REPAIR|SERVICE)|SERVICE SHOP|PHARMACY|PARLOR|CONVENIENCE|MINI.?MART|GAS(?:OLINE)? STATION|(?<!/)RETAIL (?:BLDG|BUILDING|SALES|STORE)|COFFEE|BAKERY|SUPERMARKET|GROCERY"
INDUSTRIAL = r"WAREHOUSE|MANUFACTUR|INDUSTRIAL|TILT|DISTRIBUTION|FABRICAT\w* (?:BLDG|BUILDING|SHOP)|HAZARDOUS|SELF[- ]STORAGE|STORAGE (?:BLDG|BUILDING|FACILITY)"
OFFICE = r"OFFICE|MEDICAL|CLINIC|BANK"
RETAIL = r"RETAIL|RESTAURANT|MARKET|STORE|SHOP|CAR WASH|GAS STATION|DRIVE.?THR|DEALERSHIP|CAFE|COMMERCIAL BUILDING|\bBAR\b|FOOD"


def building_type(text, sub_type):
    if re.search(MIXED_USE, text):
        return "mixed-use"
    if re.search(CARE_HOME, text):
        return "other commercial"
    if re.search(r"HOTEL|MOTEL|GUEST ?ROOMS", text) and not re.search(NOT_A_HOTEL, text):
        return "hotel"
    if sub_type == "Apartment" or re.search(APARTMENTS, text):
        return "multifamily"
    if re.search(CIVIC, text) and not re.search(r"\bWAREHOUSE\b|SELF[- ]STORAGE", text):
        return "other commercial"
    if re.search(PARKING_BUILDING, text) and not re.search(r"OFFICE|RETAIL|MEDICAL|WAREHOUSE|RESTAURANT", text):
        return "other commercial"
    if re.search(CLEARLY_RETAIL, text):
        return "retail"
    if re.search(INDUSTRIAL, text):
        return "industrial"
    if re.search(OFFICE, text):
        return "office"
    if re.search(RETAIL, text):
        return "retail"
    return "other commercial"


permits["is_revision"] = permits.permit_nbr.str.split("-").str[1].str[-4:] != "0000"
permits["is_plan_check_only"] = permits.text.str.contains(PLAN_CHECK_ONLY, regex=True)
permits["is_small_structure"] = permits.text.map(is_small_structure)
permits["building_type"] = [building_type(text, sub_type) for text, sub_type in zip(permits.text, permits.permit_sub_type)]

steps = pd.Series({
    "applications downloaded": len(permits),
    "revisions removed": int(permits.is_revision.sum()),
    "plan-check-only filings removed": int((~permits.is_revision & permits.is_plan_check_only).sum()),
    "small structures removed": int((~permits.is_revision & ~permits.is_plan_check_only & permits.is_small_structure).sum()),
})
buildings = permits[~permits.is_revision & ~permits.is_plan_check_only & ~permits.is_small_structure].copy()
steps["real building applications kept"] = len(buildings)
print(steps.to_string())
print()
print(f"City data is current to {DATA_DATE.date()}")
