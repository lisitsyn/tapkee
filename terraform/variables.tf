variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "tapkee"
}

variable "environment" {
  description = "Environment name (e.g., prod, staging, dev)"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "domain_name" {
  description = "Custom domain name for the website (optional)"
  type        = string
  default     = "tapkee.lisitsyn.me"
}

variable "create_cloudfront" {
  description = "Whether to create CloudFront distribution"
  type        = bool
  default     = true
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable access logging for S3 bucket"
  type        = bool
  default     = true
}

variable "enable_cloudfront_logging" {
  description = "Enable CloudFront access logging"
  type        = bool
  default     = true
}

variable "cloudfront_log_prefix" {
  description = "Prefix for CloudFront access logs"
  type        = string
  default     = "cloudfront-logs/"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "tapkee"
    Terraform = "true"
  }
}
