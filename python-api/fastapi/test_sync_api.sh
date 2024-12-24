curl -X GET \
  -H 'Content-Type: application/json' \
  "http://localhost:8000?name=Steve"

curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{"name": "bananas", "description": "yummy"}' \
  http://localhost:8000


curl -X PUT \
  -H 'Content-Type: application/json' \
  -d '{"name": "bananas", "description": "Delicious!"}' \
  http://localhost:8000/items

curl -X DELETE \
  -H 'Content-Type: application/json' \
  -d '{"name": "bananas", "description": "not matter"}' \
  http://localhost:8000/items