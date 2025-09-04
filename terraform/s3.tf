# S3 bucket for static website hosting
resource "aws_s3_bucket" "website" {
  bucket = local.bucket_name
  tags   = local.common_tags
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "website" {
  bucket = aws_s3_bucket.website.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

# S3 bucket public access block (only when not using CloudFront)
resource "aws_s3_bucket_public_access_block" "website" {
  count  = var.create_cloudfront ? 0 : 1
  bucket = aws_s3_bucket.website.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# S3 bucket website configuration
resource "aws_s3_bucket_website_configuration" "website" {
  bucket = aws_s3_bucket.website.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html" # SPA-style routing
  }
}

# S3 bucket policy for public read access (only when not using CloudFront)
resource "aws_s3_bucket_policy" "website_public" {
  count      = var.create_cloudfront ? 0 : 1
  bucket     = aws_s3_bucket.website.id
  depends_on = [aws_s3_bucket_public_access_block.website]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.website.arn}/*"
      }
    ]
  })
}

# S3 bucket server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "website" {
  bucket = aws_s3_bucket.website.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Optional: S3 bucket for access logs
resource "aws_s3_bucket" "logs" {
  count  = var.enable_logging ? 1 : 0
  bucket = "${local.bucket_name}-logs"
  tags   = local.common_tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  count  = var.enable_logging ? 1 : 0
  bucket = aws_s3_bucket.logs[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  count  = var.enable_logging ? 1 : 0
  bucket = aws_s3_bucket.logs[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 bucket logging configuration
resource "aws_s3_bucket_logging" "website" {
  count  = var.enable_logging ? 1 : 0
  bucket = aws_s3_bucket.website.id

  target_bucket = aws_s3_bucket.logs[0].id
  target_prefix = "access-logs/"
}
