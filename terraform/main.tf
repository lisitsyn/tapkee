terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = var.tags
  }
}

# Additional provider for ACM certificates (must be in us-east-1 for CloudFront)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  default_tags {
    tags = var.tags
  }
}

# Generate a random suffix for bucket name to ensure uniqueness
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

locals {
  bucket_name = "tapkee-lisitsyn-me"
  logs_bucket_name = "tapkee-lisitsyn-me-logs"

  # Common tags
  common_tags = merge(var.tags, {
    Environment = var.environment
  })
}
