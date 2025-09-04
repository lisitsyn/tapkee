.PHONY: help deps build static-html clean terraform-init terraform-plan terraform-deploy terraform-destroy terraform-output

# Variables
APP_PORT ?= 21600
OUTPUT_DIR = static
LEIN = lein
CURL = curl
TERRAFORM_DIR = terraform
TERRAFORM = terraform

# Help target
help:
	@echo "Available targets:"
	@echo "  deps             - Install dependencies"
	@echo "  build            - Compile the application"
	@echo "  static-html      - Generate static HTML files with guaranteed server cleanup"
	@echo "  clean            - Clean generated files and build artifacts"
	@echo "  terraform-init   - Initialize Terraform"
	@echo "  terraform-plan   - Plan Terraform deployment"
	@echo "  terraform-deploy - Deploy infrastructure using Terraform"
	@echo "  terraform-destroy- Destroy infrastructure using Terraform"
	@echo "  terraform-output - Show Terraform outputs"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  APP_PORT    - Port for the application (default: 21600)"
	@echo "  OUTPUT_DIR  - Directory for static files (default: static)"
	@echo "  TERRAFORM_DIR - Directory containing Terraform files (default: terraform)"

# Install dependencies
deps:
	$(LEIN) deps

# Build the application
build: deps
	$(LEIN) compile

# Generate static HTML files with guaranteed server cleanup
static-html: build
	@echo "Starting static HTML generation..."
	@( \
		set -e; \
		echo "Starting server on port $(APP_PORT)..."; \
		PORT=$(APP_PORT) $(LEIN) run & \
		SERVER_PID=$$!; \
		echo "Server started with PID $$SERVER_PID"; \
		echo "Waiting for server to start (timeout: 30s)..."; \
		for i in $$(seq 1 30); do \
			if curl -s --connect-timeout 1 http://localhost:$(APP_PORT) > /dev/null 2>&1; then \
				echo "Server ready after $$i seconds"; \
				break; \
			fi; \
			if [ $$i -eq 30 ]; then \
				echo "Server failed to start within 30 seconds"; \
				kill $$SERVER_PID 2>/dev/null || true; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
		echo "Cleaning and creating output directory: $(OUTPUT_DIR)"; \
		rm -rf $(OUTPUT_DIR); \
		mkdir -p $(OUTPUT_DIR); \
		echo "Generating static HTML files..."; \
		echo "  - Downloading index.html"; \
		if ! curl -s --max-time 10 http://localhost:$(APP_PORT)/ -o $(OUTPUT_DIR)/index.html; then \
			echo "Failed to download index.html"; \
			kill $$SERVER_PID 2>/dev/null || true; \
			exit 1; \
		fi; \
		echo "  - Copying static resources"; \
		cp -r resources/public/* $(OUTPUT_DIR)/; \
		echo "Static HTML generation complete!"; \
		echo "Files generated in: $(OUTPUT_DIR)/"; \
		echo "Stopping server..."; \
		kill $$SERVER_PID; \
		wait $$SERVER_PID 2>/dev/null || true; \
		echo "Server stopped"; \
	)

# Clean generated files and build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf $(OUTPUT_DIR)
	@rm -rf target
	@rm -f .server.pid
	@echo "Killing any remaining lein processes..."
	@pkill -f "lein.*run" 2>/dev/null || true
	@echo "Clean complete"

# Initialize Terraform
terraform-init:
	@echo "Initializing Terraform..."
	@cd $(TERRAFORM_DIR) && $(TERRAFORM) init
	@echo "Terraform initialization complete"

# Plan Terraform deployment
terraform-plan:
	@echo "Planning Terraform deployment..."
	@cd $(TERRAFORM_DIR) && $(TERRAFORM) plan
	@echo "Terraform plan complete"

# Deploy infrastructure using Terraform
terraform-deploy: static-html terraform-init
	@echo "Deploying infrastructure with Terraform..."
	@cd $(TERRAFORM_DIR) && $(TERRAFORM) apply -auto-approve
	@echo "Terraform deployment complete"
	@echo "Syncing static files to S3..."
	@cd $(TERRAFORM_DIR) && aws s3 sync ../$(OUTPUT_DIR)/ s3://$$($(TERRAFORM) output -raw s3_bucket_name)/ --delete
	@echo "Static files synced successfully"
	@echo ""
	@echo "Deployment complete! Your website should be available at:"
	@cd $(TERRAFORM_DIR) && echo "  CloudFront URL: $$($(TERRAFORM) output -raw cloudfront_domain_name)"
	@cd $(TERRAFORM_DIR) && if $(TERRAFORM) output domain_name > /dev/null 2>&1; then echo "  Custom Domain: $$($(TERRAFORM) output -raw domain_name)"; fi

# Destroy infrastructure using Terraform
terraform-destroy:
	@echo "WARNING: This will destroy all infrastructure!"
	@echo "Press Ctrl+C within 10 seconds to cancel..."
	@sleep 10
	@echo "Destroying infrastructure with Terraform..."
	@cd $(TERRAFORM_DIR) && $(TERRAFORM) destroy -auto-approve
	@echo "Infrastructure destroyed"

# Show Terraform outputs
terraform-output:
	@echo "Terraform outputs:"
	@cd $(TERRAFORM_DIR) && $(TERRAFORM) output
