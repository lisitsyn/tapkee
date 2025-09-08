.PHONY: help deps build static-html clean terraform-init terraform-plan terraform-deploy terraform-destroy terraform-output

# Variables
OUTPUT_DIR = static
LEIN = lein
TERRAFORM_DIR = terraform
TERRAFORM = terraform

# Help target
help:
	@echo "Available targets:"
	@echo "  deps             - Install dependencies"
	@echo "  build            - Compile the application"
	@echo "  static-html      - Generate static HTML files"
	@echo "  clean            - Clean generated files and build artifacts"
	@echo "  terraform-init   - Initialize Terraform"
	@echo "  terraform-plan   - Plan Terraform deployment"
	@echo "  terraform-deploy - Deploy infrastructure using Terraform"
	@echo "  terraform-destroy- Destroy infrastructure using Terraform"
	@echo "  terraform-output - Show Terraform outputs"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  OUTPUT_DIR  - Directory for static files (default: static)"
	@echo "  TERRAFORM_DIR - Directory containing Terraform files (default: terraform)"

# Install dependencies
deps:
	$(LEIN) deps

# Build the application
build: deps
	$(LEIN) compile

# Generate static HTML files
static-html: build
	@echo "Starting static HTML generation..."
	@echo "Cleaning and creating output directory: $(OUTPUT_DIR)"
	@rm -rf $(OUTPUT_DIR)
	@mkdir -p $(OUTPUT_DIR)
	@echo "Generating index.html..."
	@$(LEIN) run > $(OUTPUT_DIR)/index.html
	@echo "Copying static resources..."
	@cp -r resources/public/* $(OUTPUT_DIR)/
	@echo "Static HTML generation complete!"
	@echo "Files generated in: $(OUTPUT_DIR)/"

local: static-html
	cd $(OUTPUT_DIR) && python3 -m http.server

# Clean generated files and build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf $(OUTPUT_DIR)
	@rm -rf target
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
