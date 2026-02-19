# Data Directory

## Structure
- `raw/`: Place the original EMSCAD dataset here
- `processed/`: Processed datasets after annotation and balancing

## Dataset Format
The EMSCAD dataset should be in CSV format with the following columns:
- title
- location
- department
- salary_range
- company_profile
- description
- requirements
- benefits
- telecommuting
- has_company_logo
- has_questions
- employment_type
- required_experience
- required_education
- industry
- function
- fraudulent

## Note
The dataset will be automatically annotated into 4 types during processing.