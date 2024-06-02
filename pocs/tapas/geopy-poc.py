from geopy.geocoders import Nominatim

# Initialize Nominatim API with a unique user_agent
geolocator = Nominatim(user_agent="geocoder-poc")

# Example latitude and longitude
latitude = "48.8588443"
longitude = "2.2943506"

# Perform reverse geocoding
location = geolocator.reverse((latitude, longitude), exactly_one=True)

# Extract and print address information
address = location.address
print(f"Address: {address}")
