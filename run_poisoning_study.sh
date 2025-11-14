#!/bin/bash

###############################################################################
# Data Poisoning Study - Complete Execution Script
# 
# This script runs the entire data poisoning study:
# 1. Trains models with various poisoning levels
# 2. Analyzes and visualizes results
# 3. Validates model performance
# 4. Generates comprehensive reports
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         DATA POISONING STUDY - IRIS DATASET                    ║"
echo "║         MLflow Experiment Tracking & Analysis                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"


# Step 1: Pull data from DVC
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}STEP 1: Pulling dataset from DVC${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

dvc pull data/data.csv.dvc
echo -e "${GREEN}✅ Data pulled successfully${NC}\n"

# Step 2: Train models with poisoning
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}STEP 2: Training models with data poisoning${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${YELLOW}This will train 10 models (5 poison levels × 2 poison types)${NC}"
echo -e "${YELLOW}Expected duration: 2-5 minutes${NC}\n"

python3 train_with_poisoning.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Training completed successfully${NC}\n"
else
    echo -e "\n${RED}❌ Training failed${NC}\n"
    exit 1
fi

# Step 3: Analyze results
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}STEP 3: Analyzing results and generating visualizations${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

python3 tests/analyze_poisoning_results.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Analysis completed${NC}\n"
else
    echo -e "\n${RED}❌ Analysis failed${NC}\n"
    exit 1
fi

# Step 4: Run validation tests
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}STEP 4: Running validation tests${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

if ! pytest tests/test_with_poisoning.py -v -s; then
    PYTEST_EXIT_CODE=$?
    echo -e "\n${RED}⚠️ Pytest reported failures (this may be expected for a poisoning study).${NC}"
else
    PYTEST_EXIT_CODE=0
    echo -e "\n${GREEN}✅ Pytest passed.${NC}"
fi

PYTEST_EXIT_CODE=$?

# Step 5: Generate summary report
echo -e "\n${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}STEP 5: Generating final report${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Create output directory
mkdir -p poisoning_study_results
mv -f poisoning_analysis_dashboard.png poisoning_study_results/ 2>/dev/null || true
mv -f poisoning_summary.csv poisoning_study_results/ 2>/dev/null || true
mv -f confusion_matrix_*.png poisoning_study_results/ 2>/dev/null || true
mv -f cml_report_poisoning.md poisoning_study_results/ 2>/dev/null || true
mv -f metrics_poisoning.json poisoning_study_results/ 2>/dev/null || true
mv -f data_validation_report.json poisoning_study_results/ 2>/dev/null || true
mv -f robustness_metrics.json poisoning_study_results/ 2>/dev/null || true

# Create index file
cat > poisoning_study_results/README.md << 'EOF'
# Data Poisoning Study Results

## 📁 Files Generated

- `poisoning_analysis_dashboard.png` - Comprehensive visualization of poisoning impact
- `poisoning_summary.csv` - Tabular summary of all experiment runs
- `confusion_matrix_*.png` - Confusion matrices for each poisoning level
- `cml_report_poisoning.md` - Markdown report for CI/CD integration
- `metrics_poisoning.json` - JSON metrics for automated processing
- `data_validation_report.json` - Data quality assessment results
- `robustness_metrics.json` - Model robustness test results

## 🔗 View in MLflow

Access the MLflow UI to explore all experiment runs interactively:
- Experiment: `iris_data_poisoning`
- Compare runs side-by-side
- Download models and artifacts
- View detailed metrics and parameters

## 📊 Key Findings

Review `poisoning_analysis_dashboard.png` for visual insights into:
1. How accuracy degrades with increasing poisoning levels
2. Comparison between feature and label poisoning
3. Train vs test accuracy gaps indicating data quality issues
4. F1 score trends across poisoning levels

## 🛡️ Next Steps

1. Review the validation test results
2. Check if any models failed quality thresholds
3. Document lessons learned
4. Update production data validation pipelines
EOF

echo -e "${GREEN}✅ Results organized in ${BOLD}poisoning_study_results/${NC}\n"

# Final summary
echo -e "${BLUE}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║                    STUDY COMPLETE                              ║${NC}"
echo -e "${BLUE}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${GREEN}📊 Generated Artifacts:${NC}"
echo -e "   • Training logs and metrics in MLflow"
echo -e "   • Visualization dashboard"
echo -e "   • Confusion matrices for all runs"
echo -e "   • Performance comparison tables"
echo -e "   • Data validation reports"
echo -e "   • Test results\n"

echo -e "${YELLOW}📁 All results saved to: ${BOLD}poisoning_study_results/${NC}\n"

if [ -f ".env" ]; then
    source .env
    echo -e "${BLUE}🔗 View in MLflow UI:${NC}"
    echo -e "   ${BOLD}${MLFLOW_TRACKING_SERVER}${NC}"
    echo -e "   ${YELLOW}Experiment: iris_data_poisoning${NC}\n"
fi

echo -e "${GREEN}📖 For detailed explanation, see:${NC}"
echo -e "   ${BOLD}DATA_POISONING_GUIDE.md${NC}\n"

# Exit with pytest status
if [ $PYTEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ ALL TESTS PASSED - Models meet quality thresholds${NC}\n"
    exit 0
else
    echo -e "${RED}${BOLD}⚠️  SOME TESTS FAILED - Review validation results${NC}\n"
    exit $PYTEST_EXIT_CODE
fi