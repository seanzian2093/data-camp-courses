import redis

# Create a connection to Redis cluster
redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Store the city key-value pair
redis_conn.set("city", "London")

# Store the sunshine key-value pair
redis_conn.set("sunshine", "7")

# Retrieve values stored at the city and sunshine keys
city = redis_conn.get("city")
sunshine = redis_conn.get("sunshine")

print(city)
print(sunshine)

# Create a dictionary containing weather data
london_weather_mapping = {"temperature": 42, "humidity": 88, "visibility": "low"}

# Store the london_weather key-value pair
redis_conn.hset("london_weather", mapping=london_weather_mapping)

# Retrieve and print the london_weather key-value pair
print(redis_conn.hgetall("london_weather"))
