.PHONY: help deps build static-html generate-preview clean terraform-init terraform-plan terraform-deploy terraform-destroy terraform-output

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
	@echo "  generate-preview - Generate preview image for social media"
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
	@echo "Generating preview image..."
	@$(MAKE) generate-preview
	@echo "Static HTML generation complete!"
	@echo "Files generated in: $(OUTPUT_DIR)/"

# Generate preview image for social media
generate-preview:
	@echo "Generating preview image..."
	@if command -v node >/dev/null 2>&1; then \
		if [ -f package.json ]; then \
			echo "Installing Node.js dependencies..."; \
			npm install; \
			echo "Running preview generation script..."; \
			node render_preview.js; \
			echo "Preview image generated successfully!"; \
		else \
			echo "Error: package.json not found. Please ensure Node.js dependencies are configured."; \
			exit 1; \
		fi; \
	else \
		echo "Warning: Node.js not found. Skipping preview generation."; \
		echo "To generate preview images, please install Node.js and run 'make generate-preview'"; \
	fi

local: static-html
	cd $(OUTPUT_DIR) && python3 -m http.server

# Clean generated files and build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf $(OUTPUT_DIR)
	@rm -rf target
	@rm -rf node_modules
	@rm -f resources/public/img/tapkee-preview.png
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
	@echo "Invalidating CloudFront cache..."
	@cd $(TERRAFORM_DIR) && aws cloudfront create-invalidation --distribution-id $$($(TERRAFORM) output -raw cloudfront_distribution_id) --paths "/*"
	@echo "CloudFront cache invalidated successfully"
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
