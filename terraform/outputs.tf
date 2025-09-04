# S3 bucket outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.website.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.website.arn
}

output "s3_website_endpoint" {
  description = "S3 website endpoint"
  value       = aws_s3_bucket_website_configuration.website.website_endpoint
}

output "s3_website_domain" {
  description = "S3 website domain"
  value       = aws_s3_bucket_website_configuration.website.website_domain
}

# CloudFront outputs
output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = var.create_cloudfront ? aws_cloudfront_distribution.website[0].id : null
}

output "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN"
  value       = var.create_cloudfront ? aws_cloudfront_distribution.website[0].arn : null
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = var.create_cloudfront ? aws_cloudfront_distribution.website[0].domain_name : null
}

# Route53 outputs
output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = var.domain_name != "" ? aws_route53_zone.main[0].zone_id : null
}

output "route53_name_servers" {
  description = "Route53 hosted zone name servers"
  value       = var.domain_name != "" ? aws_route53_zone.main[0].name_servers : null
}

output "certificate_arn" {
  description = "ACM certificate ARN"
  value       = var.domain_name != "" ? aws_acm_certificate.cert[0].arn : null
}

# Website URLs
output "website_url" {
  description = "Website URL"
  value = var.domain_name != "" ? "https://${var.domain_name}" : (
    var.create_cloudfront ? "https://${aws_cloudfront_distribution.website[0].domain_name}" :
    "http://${aws_s3_bucket_website_configuration.website.website_endpoint}"
  )
}

output "s3_sync_command" {
  description = "AWS CLI command to sync files to S3"
  value       = "aws s3 sync ./static/ s3://${aws_s3_bucket.website.bucket}/ --delete"
}

output "cloudfront_invalidation_command" {
  description = "AWS CLI command to invalidate CloudFront cache"
  value       = var.create_cloudfront ? "aws cloudfront create-invalidation --distribution-id ${aws_cloudfront_distribution.website[0].id} --paths '/*'" : null
}
