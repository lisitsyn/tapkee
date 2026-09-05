# Route53 hosted zone (optional, only if domain_name is provided)
# This will create a new hosted zone for your domain
resource "aws_route53_zone" "main" {
  count = var.domain_name != "" ? 1 : 0
  name  = var.domain_name
  tags  = local.common_tags
}

# Use the created hosted zone (or comment out the resource above and uncomment this data source if you have an existing hosted zone)
# data "aws_route53_zone" "main" {
#   count = var.domain_name != "" ? 1 : 0
#   name  = var.domain_name
# }

# ACM Certificate for HTTPS (must be in us-east-1 for CloudFront)
resource "aws_acm_certificate" "cert" {
  count             = var.domain_name != "" ? 1 : 0
  domain_name       = var.domain_name
  validation_method = "DNS"

  # Temporarily removed wildcard to simplify certificate request
  # subject_alternative_names = [
  #   "*.${var.domain_name}"
  # ]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags

  # ACM certificates for CloudFront must be in us-east-1
  provider = aws.us_east_1
}

# DNS validation records for ACM certificate
resource "aws_route53_record" "cert_validation" {
  for_each = var.domain_name != "" ? {
    for dvo in aws_acm_certificate.cert[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  zone_id = aws_route53_zone.main[0].zone_id
  name    = each.value.name
  type    = each.value.type
  records = [each.value.record]
  ttl     = 60
}

# ACM certificate validation
resource "aws_acm_certificate_validation" "cert" {
  count           = var.domain_name != "" ? 1 : 0
  certificate_arn = aws_acm_certificate.cert[0].arn
  validation_record_fqdns = [
    for record in aws_route53_record.cert_validation : record.fqdn
  ]

  provider = aws.us_east_1

  timeouts {
    create = "5m"
  }
}

# Route53 A record pointing to CloudFront distribution
resource "aws_route53_record" "website" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.website[0].domain_name
    zone_id                = aws_cloudfront_distribution.website[0].hosted_zone_id
    evaluate_target_health = false
  }
}

# Route53 AAAA record for IPv6 support
resource "aws_route53_record" "website_ipv6" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.website[0].domain_name
    zone_id                = aws_cloudfront_distribution.website[0].hosted_zone_id
    evaluate_target_health = false
  }
}
