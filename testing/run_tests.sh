#!/bin/bash

# Test Suite Runner for BDI-ToM Simulation
# ==========================================
# This script generates test data and runs validation tests

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_RUN_ID=999
TEST_NUM_AGENTS=5
TEST_NUM_TRAJECTORIES=25
TEST_DATA_DIR="data/simulation_data/run_${TEST_RUN_ID}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BDI-ToM Simulation Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Clean up any existing test data
echo -e "${YELLOW}[1/3] Cleaning up previous test data...${NC}"
if [ -d "$TEST_DATA_DIR" ]; then
    rm -rf "$TEST_DATA_DIR"
    echo -e "${GREEN}✓ Removed old test data${NC}"
else
    echo -e "${GREEN}✓ No previous test data found${NC}"
fi
echo ""

# Step 2: Generate test simulation data
echo -e "${YELLOW}[2/3] Generating test simulation data...${NC}"
echo -e "  - Agents: ${TEST_NUM_AGENTS}"
echo -e "  - Trajectories per agent: ${TEST_NUM_TRAJECTORIES}"
echo -e "  - Run ID: ${TEST_RUN_ID}"
echo ""

python simulation_controller/simulation_runner.py \
    -n "$TEST_NUM_AGENTS" \
    -m "$TEST_NUM_TRAJECTORIES" \
    -x "$TEST_RUN_ID" \
    --quiet

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Test data generated successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Failed to generate test data${NC}"
    exit 1
fi
echo ""

# Step 3: Run validation tests
echo -e "${YELLOW}[3/3] Running validation tests...${NC}"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Verify goal nodes are reached
echo -e "${BLUE}Running: test_goal_reached.py${NC}"
python testing/test_goal_reached.py --run_id "$TEST_RUN_ID"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: Goal reached validation${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED: Goal reached validation${NC}"
    ((TESTS_FAILED++))
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

# Cleanup: Delete test data
echo -e "${YELLOW}Cleaning up test data...${NC}"
if [ -d "$TEST_DATA_DIR" ]; then
    rm -rf "$TEST_DATA_DIR"
    echo -e "${GREEN}✓ Test data deleted${NC}"
else
    echo -e "${YELLOW}⚠ Test data directory not found${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
