# Makefile for Q3Nexus CUDA Operator

# Configuration
PYTHON ?= python
PACKAGE_NAME := Q3Nexus-Operator
MODULE_NAME := Q3Nexus_Ops
BUILD_DIRS := build dist *.egg-info
TEST_SCRIPT := test_install.py

.PHONY: all install clean uninstall

all: install clean

install:
	@echo "\n\033[34m=== Building CUDA Extension ===\033[0m"
	@$(PYTHON) setup.py install 2>&1 | tee build.log
	@echo "\n\033[34m=== Testing Installation ===\033[0m"
	@echo "from $(MODULE_NAME) import rmsnorm_bf16xbf16" >> $(TEST_SCRIPT)
	@echo "import torch" > $(TEST_SCRIPT)
	@echo "import $(MODULE_NAME)" > $(TEST_SCRIPT)
	@$(PYTHON) $(TEST_SCRIPT)
	@rm -f $(TEST_SCRIPT)
	@echo "\033[32m✓ Successfully installed $(PACKAGE_NAME)\033[0m"

clean:
	@echo "\n\033[34m=== Cleaning Build Files ===\033[0m"
	@rm -rf $(BUILD_DIRS) build.log
	@rm -rf python/Q3Nexus_Operator.egg-info/
	@echo "\033[32m✓ Build directories removed\033[0m"

uninstall:
	@echo "\n\033[34m=== Uninstalling Package ===\033[0m"
	@if pip show $(PACKAGE_NAME) >/dev/null 2>&1; then \
		pip uninstall -y $(PACKAGE_NAME); \
		echo "\033[32m✓ Package uninstalled\033[0m"; \
	else \
		echo "\033[33mPackage not found in pip, trying manual cleanup...\033[0m"; \
		rm -rf $(INSTALLED_FILES) 2>/dev/null || true; \
		find $(shell python -c "import site; print(site.getsitepackages()[0])") -name "$(MODULE_NAME)*" -exec rm -rf {} + 2>/dev/null || true; \
		echo "\033[32m✓ Manual cleanup completed\033[0m"; \
	fi
	@$(MAKE) clean