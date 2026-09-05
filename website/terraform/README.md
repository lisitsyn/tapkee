# Tapkee Website Terraform Infrastructure

This directory contains Terraform configuration files to deploy the Tapkee website as a static website on AWS using S3, CloudFront, and optionally Route53.

## Architecture

The infrastructure includes:

- **S3 Bucket**: Hosts the static website files
- **CloudFront Distribution**: CDN for global content delivery and HTTPS
- **CloudFront Logging**: Access logs stored in dedicated S3 bucket (optional)
- **Route53 Records**: Custom domain support (optional)
- **ACM Certificate**: SSL/TLS certificate for HTTPS (optional)

## Prerequisites

1. **AWS CLI**: Install and configure with your credentials
   ```bash
   aws configure
   ```

2. **Terraform**: Install Terraform (>= 1.0)
   ```bash
   # macOS
   brew install terraform
   
   # Or download from https://terraform.io/downloads
   ```

3. **Build Tools**: Ensure you have `make` and other build dependencies for the Clojure project

## Quick Start

1. **Configure Variables**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your specific configuration
   ```

2. **Deploy Everything**:
   ```bash
   ./deploy.sh
   ```

This will:
- Build the static website
- Initialize Terraform
- Plan the infrastructure
- Deploy to AWS
- Sync files to S3
- Invalidate CloudFront cache (if enabled)

## Configuration Options

### terraform.tfvars

```hcl
project_name      = "tapkee"
environment       = "prod"
aws_region        = "us-east-1"
domain_name       = "tapkee.example.com"  # Optional
create_cloudfront = true
enable_versioning        = true
enable_logging           = true
enable_cloudfront_logging = true
cloudfront_log_prefix    = "cloudfront-logs/"

tags = {
  Project     = "tapkee"
  Environment = "prod"
  Terraform   = "true"
  Owner       = "your-name"
}
```

### Key Variables

- `domain_name`: Your custom domain (leave empty for CloudFront/S3 URLs only)
- `create_cloudfront`: Set to `false` to use S3 website hosting only
- `enable_versioning`: Enable S3 object versioning
- `enable_logging`: Create separate S3 bucket for S3 access logs
- `enable_cloudfront_logging`: Enable CloudFront access logging to S3
- `cloudfront_log_prefix`: Prefix for CloudFront log files in S3

## Deployment Scenarios

### 1. S3 Only (No CloudFront)

```hcl
create_cloudfront = false
domain_name       = ""
```

This creates a simple S3 static website (HTTP only).

### 2. S3 + CloudFront (No Custom Domain)

```hcl
create_cloudfront = true
domain_name       = ""
```

This creates S3 + CloudFront with HTTPS using CloudFront's domain.

### 3. Full Setup with Custom Domain

```hcl
create_cloudfront = true
domain_name       = "tapkee.example.com"
```

This creates the full stack with your custom domain and SSL certificate.

**Note**: For custom domains, you need a Route53 hosted zone for your domain.

## Manual Commands

### Initialize Terraform
```bash
terraform init
```

### Plan Deployment
```bash
terraform plan
```

### Apply Infrastructure
```bash
terraform apply
```

### Build and Sync Files Only
```bash
./deploy.sh build
./deploy.sh sync
```

### Invalidate CloudFront Cache
```bash
./deploy.sh invalidate
```

## File Structure

```
terraform/
├── main.tf              # Provider and general configuration
├── variables.tf         # Input variables
├── s3.tf               # S3 bucket configuration
├── cloudfront.tf       # CloudFront distribution
├── route53.tf          # DNS and SSL certificate
├── outputs.tf          # Output values
├── deploy.sh           # Deployment script
├── terraform.tfvars.example  # Example configuration
└── README.md          # This file
```

## Outputs

After deployment, Terraform provides useful outputs:

```bash
terraform output website_url
terraform output s3_bucket_name
terraform output cloudfront_domain_name
terraform output cloudfront_logs_bucket_name
terraform output s3_sync_command
terraform output cloudfront_invalidation_command
```

## Cost Optimization

### S3 Costs
- Standard storage: ~$0.023/GB/month
- Requests: ~$0.0004/1000 GET requests

### CloudFront Costs
- Data transfer: ~$0.085/GB for first 10TB
- Requests: ~$0.0075/10,000 HTTPS requests
- Price class set to `PriceClass_100` (US, Canada, Europe only)

### Route53 Costs
- Hosted zone: $0.50/month
- DNS queries: $0.40/million queries

## Security Features

- S3 bucket encryption enabled
- CloudFront Origin Access Control (OAC)
- HTTPS redirect enforced
- Security headers can be added via Lambda@Edge (not included)

## Troubleshooting

### Common Issues

1. **Certificate validation timeout**: 
   - Ensure Route53 hosted zone exists for your domain
   - Check DNS propagation

2. **S3 sync permission denied**:
   - Verify AWS credentials have S3 permissions
   - Check bucket policy

3. **CloudFront not updating**:
   - Run cache invalidation
   - Wait for distribution deployment (5-15 minutes)

### Useful Commands

```bash
# Check AWS credentials
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# Check CloudFront distributions
aws cloudfront list-distributions

# Validate Terraform configuration
terraform validate

# Show current state
terraform show
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all resources including the S3 bucket and its contents.

## Contributing

When modifying the Terraform configuration:

1. Run `terraform fmt` to format code
2. Run `terraform validate` to validate syntax
3. Test with `terraform plan` before applying
4. Update this README if adding new features
