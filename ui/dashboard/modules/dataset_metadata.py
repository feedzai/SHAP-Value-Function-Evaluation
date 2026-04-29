# modules/dataset_metadata.py

DATASET_METADATA = {
    "GermanCredit": {
        "title": "UCI German Credit Risk",
        "goal": (
            "You will review anonymized credit applications and determine whether "
            "the applicant should be classified as *Risk* or *No Risk* based on the "
            "feature information and model explanations provided."
        ),
        "label_definition": (
            "- **1 = Risk**: Applicant shows indicators of likely default or financial "
            "instability. These applicants may have poor credit history, unstable "
            "employment, or other risk factors.\n"
            "- **0 = No Risk**: Applicant demonstrates stable financial behavior and is"
            " expected to repay the credit normally based on their profile."
        ),
        "description": (
            "The UCI German Credit dataset contains 1,000 credit applications with "
            "20 attributes including demographic, financial, and employment "
            "information. This dataset is widely used in credit scoring research and "
            "fairness studies. Note: The dataset uses Deutsche Mark (DM) currency and "
            "reflects historical German banking practices."
        ),
        "feature_definitions": {
            "Account Status": {
                "definition": ("Status of existing checking account."),
                "categories": [
                    "<0 DM",
                    "0-200 DM",
                    "\\>200 DM or salary assignments for at least 1 year",
                    "no checking account",
                ],
            },
            "Duration (months)": {
                "definition": "Duration of the requested credit in months.",
            },
            "Credit History": {
                "definition": "Previous credit payment behavior.",
                "categories": [
                    "no credits taken / all credits paid back duly",
                    "all credits at this bank paid back duly",
                    "existing credits paid back duly till now",
                    "delay in paying off in the past",
                    "critical account/other credits existing (not at this bank)",
                ],
            },
            "Purpose": {
                "definition": "Purpose of the credit request.",
                "categories": [
                    "car (new)",
                    "car (used)",
                    "furniture/equipment",
                    "radio/television",
                    "domestic appliances",
                    "repairs",
                    "education",
                    "vacation",
                    "retraining",
                    "business",
                    "other",
                ],
            },
            "Credit Amount": {
                "definition": "Amount of credit requested.",
            },
            "Savings Account": {
                "definition": "Status of savings account/bonds.",
                "categories": [
                    "<100 DM",
                    "100-500 DM",
                    "500-1000 DM",
                    "\\>1000 DM",
                    "unknown/no savings account",
                ],
            },
            "Employed Since": {
                "definition": "Duration of current employment.",
                "categories": [
                    "unemployed",
                    "<1 year",
                    "1-4 years",
                    "4-7 years",
                    "≥7 years",
                ],
            },
            "Installment Rate (percentage of disposable income)": {
                "definition": (
                    "Monthly installment payment as a percentage of disposable income."
                ),
            },
            "Personal Status and Sex": {
                "definition": "Combined sex and marital status indicator.",
                "categories": [
                    "male: divorced/separated",
                    "female: divorced/separated/married",
                    "male: single",
                    "male: married/widowed",
                    "female: single",
                ],
            },
            "Other Debtors/Guarantors": {
                "definition": "Presence of co-applicants or guarantors.",
                "categories": [
                    "none",
                    "co-applicant",
                    "guarantor",
                ],
            },
            "Present Residence Since": {
                "definition": "Duration of residence at current address (in years).",
            },
            "Property": {
                "definition": "Type of property owned by the applicant.",
                "categories": [
                    "real estate",
                    "building society savings agreement/life insurance",
                    "car or other",
                    "unknown/no property",
                ],
            },
            "Age (years)": {
                "definition": "Age of the credit applicant in years.",
            },
            "Other Installment Plans": {
                "definition": "Existing installment plans with other institutions.",
                "categories": [
                    "bank",
                    "stores",
                    "none",
                ],
            },
            "Housing": {
                "definition": "Type of housing arrangement.",
                "categories": [
                    "rent",
                    "own",
                    "for free",
                ],
            },
            "Number of Credits": {
                "definition": "Number of existing credits at this bank.",
            },
            "Job": {
                "definition": "Employment category and skill level.",
                "categories": [
                    "unemployed/unskilled - non-resident",
                    "unskilled - resident",
                    "skilled employee/official",
                    "management/self-employed/highly qualified employee/officer",
                ],
            },
            "Number of Dependents": {
                "definition": (
                    "Number of people the applicant is legally obligated "
                    "to support financially."
                ),
            },
            "Telephone": {
                "definition": "Whether the applicant has a telephone.",
            },
            "Foreign Worker": {
                "definition": "Whether the applicant is a foreign worker.",
            },
        },
    },
    "MaternalRisk": {
        "title": "Maternal Health Risk Assessment",
        "goal": (
            "You will evaluate pregnant patient profiles and classify the level of "
            "*Health Risk* based on clinical indicators."
        ),
        "label_definition": (
            "Pregnant patients are classified as *High Risk* or *Low Risk* based "
            "on clinical indicators."
        ),
        "description": (
            "Data has been collected from different hospitals, community clinics, "
            "maternal health cares through the IoT based risk monitoring system. The "
            "goal is to understand which health conditions are the strongest "
            "indications for health risks during pregnancy."
        ),
        "feature_definitions": {
            "Age": {
                "definition": "Age in years when a woman is pregnant.",
            },
            "SystolicBP": {
                "definition": "Upper value of Blood Pressure in mmHg.",
            },
            "DiastolicBP": {
                "definition": "Lower value of Blood Pressure in mmHg.",
            },
            "BS": {
                "definition": "Blood glucose levels is in terms of a molar concentration, mmol/L."
            },
            "HeartRate": {
                "definition": "A normal resting heart rate in beats per minute."
            },
        },
    },
    "HELOC": {
        "title": "FICO HELOC Creditworthiness",
        "goal": (
            "You will assess anonymized borrower credit profiles and determine whether "
            "they represent a *Good* or *Bad* applicant for a home equity line of credit."
        ),
        "label_definition": (
            "- **1 = Bad**: Applicant is more likely to become delinquent.\n"
            "- **0 = Good**: Applicant is expected to maintain repayment performance."
        ),
        "description": (
            "Each entry in the dataset is a line of credit, typically offered by a bank"
            " as a percentage of home equity (the difference between the current market"
            " value of a home and its purchase price). The customers in this dataset "
            "have requested a credit line in the range of \\$5,000 - \\$150,000. The "
            "fundamental task is to use the information about the applicant in their "
            "credit report to predict whether they will repay their HELOC account "
            "within 2 years."
        ),
        "feature_definitions": {
            "Estimate of risk": {"definition": "The estimated risk of the applicant."},
            "Months since first trade": {
                "definition": "The number of months since the first trade."
            },
            "Months since last trade": {
                "definition": "The number of months since the last trade."
            },
            "Average duration of resolution": {
                "definition": "The average duration of the resolution of the trades."
            },
            "Number of satisfactory trades": {
                "definition": "The number of satisfactory trades."
            },
            "Number of trades insolvent for >60 days": {
                "definition": "The number of trades insolvent for more than 60 days."
            },
            "Number of trades insolvent for >90 days": {
                "definition": "The number of trades insolvent for more than 90 days."
            },
            "Percentage of legal trades": {
                "definition": "The percentage of legal trades."
            },
            "Months since last illegal trade": {
                "definition": "The number of months since the last illegal trade."
            },
            "Maximum illegal trades over last year": {
                "definition": "The maximum number of illegal trades over the last year."
            },
            "Maximum number of illegal trades": {
                "definition": "The maximum number of illegal trades."
            },
            "Total number of trades": {"definition": "The total number of trades."},
            "Number of trades initiated in last year": {
                "definition": "The number of trades initiated in the last year."
            },
            "Percentage of installment trades": {
                "definition": "The percentage of installment trades."
            },
            "Months since last inquiry excluding recent": {
                "definition": (
                    "The number of months since the last inquiry excluding "
                    "the recent one."
                )
            },
            "Number of inquiries in last 6 months": {
                "definition": "The number of inquiries in the last 6 months."
            },
            "Number of inquiries in last 6 months excluding recent": {
                "definition": (
                    "The number of inquiries in the last 6 months excluding "
                    "the most recent ones, i.e. those in the past 7 days."
                )
            },
            "Net fraction of revolving burden": {
                "definition": "The net fraction of revolving burden."
            },
            "Net fraction of installment burden": {
                "definition": "The net fraction of installment burden."
            },
            "Number of revolving trades with balance": {
                "definition": "The number of revolving trades with balance."
            },
            "Number of installment trades with balance": {
                "definition": "The number of installment trades with balance."
            },
            "Number of banks with high ratio": {
                "definition": "The number of banks with high utilization ratio."
            },
            "Percentage of trades with balance": {
                "definition": "The percentage of trades with balance."
            },
        },
    },
    "Adult": {
        "title": "Adult Income (UCI Census)",
        "goal": (
            "You will classify whether an individual is at risk of poverty based on "
            "their annual income based on demographic and occupational attributes."
        ),
        "label_definition": (
            "- **1 = ≤50K**: Predicted annual income under $50,000, risk of poverty.\n"
            "- **0 = >50K**: Predicted annual income over $50,000, no risk of poverty."
        ),
        "description": (
            "An individual's annual income results from various factors. Intuitively, "
            "it is influenced by the individual's education level, age, gender, "
            "occupation, etc. We can explore the possibility in predicting income level"
            "based on the individual's personal information."
        ),
        "feature_definitions": {
            "Class of Worker": {
                "definition": "Class of worker.",
                "categories": [
                    "Employee of a private for-profit",
                    "Employee of a private not-for-profit",
                    "Local government employee",
                    "State government employee",
                    "Federal government employee",
                    "Self-employed-not incorporated",
                    "Self-employed-incorporated",
                    "Working without pay",
                    "Unemployed",
                ],
            },
            "Marital Status": {
                "definition": "Marital status.",
                "categories": [
                    "Married",
                    "Widowed",
                    "Divorced",
                    "Separated",
                    "Never married",
                ],
            },
            "Occupation Code": {
                "definition": "Occupation code.",
                "categories": [
                    (
                        "Please see ACS PUMS documentation for the full list of "
                        "occupation codes."
                    ),
                ],
            },
            "Place of Birth Code": {
                "definition": "Place of birth code.",
                "categories": [
                    (
                        "Range of values includes most countries and individual "
                        "US states; please see ACS PUMS documentation for the full list."
                    ),
                ],
            },
            "Relationship": {
                "definition": "Relationship.",
                "categories": [
                    "Reference person",
                    "Husband/wife",
                    "Biological child",
                    "Adopted child",
                    "Stepchild",
                    "Brother/sister",
                    "Father/mother",
                    "Grandchild",
                    "Parent-in-law",
                    "Child-in-law",
                    "Other relative",
                    "Roomer",
                    "Housemate",
                    "Unmarried partner",
                    "Foster child",
                    "Other non-relative",
                    "Institutionalized group quarters population",
                    "Noninstitutionalized group quarters population",
                ],
            },
            "Race": {
                "definition": "Race.",
                "categories": [
                    "White",
                    "Black/African American",
                    "American Indian",
                    "Alaska Native",
                    "American Indian/Alaska Native",
                    "Asian",
                    "Native Hawaiian/Other Pacific Islander",
                    "Other Race alone",
                    "Two major races",
                ],
            },
            "Sex": {"definition": "Sex.", "categories": ["Male", "Female"]},
            "Education": {
                "definition": "Education.",
                "categories": [
                    "No schooling",
                    "Nursery school",
                    "Kindergarten",
                    "Grade 1",
                    "Grade 2",
                    "Grade 3",
                    "Grade 4",
                    "Grade 5",
                    "Grade 6",
                    "Grade 7",
                    "Grade 8",
                    "Grade 9",
                    "Grade 10",
                    "Grade 11",
                    "Grade 12 - no diploma",
                    "High school diploma",
                    "GED or alternative credential",
                    "Some college, but no degree",
                    "\\>1 years college credit, no degree",
                    "Associate's degree",
                    "Bachelor's degree",
                    "Master's degree",
                    "Professional degree",
                    "Doctorate degree",
                ],
            },
            "Work Hours per week": {"definition": "Work hours per week."},
            "Age": {"definition": "Age in years."},
        },
    },
}
