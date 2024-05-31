from geopy.geocoders import Nominatim

# Initialize Nominatim API with a unique user_agent
geolocator = Nominatim(user_agent="geopy-poc-ipcc-chatbot")

# Example latitude and longitude
latitude = "-88.75"
longitude = "6.25"

# Perform reverse geocoding
location = geolocator.reverse((latitude, longitude), exactly_one=True)

# Extract and print address information
print(f"Location: {location}")
address = location.address
print(f"Address: {address}")
