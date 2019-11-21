from django.db import models



{
  "access_token": "ya29.Il-xB7WVKOHJzXUIBm-8VXo9JJyiIjdCyhhMNoMrLEXElxcsNwldI8UyF4O048S0fCfMh_H8cOF5GAGDajWVZ_KVHTPVIe4gAzw_L0M5XSGIw8O4P_HFh9-U4Op_l9Fx_Q", 
  "scope": "https://www.googleapis.com/auth/cloud-platform", 
  "token_type": "Bearer", 
  "expires_in": 3599, 
  "refresh_token": "1//04IoEK_XZQRl4CgYIARAAGAQSNwF-L9Ir8Yty-QU3X8UHWdtqwhyc0Vk_tEg9yg4ymbxLwA5vqv2czKycqk8db0z2a9RJYrYd8R4"
}


class Segment(models.Model):
    text = models.TextField()
