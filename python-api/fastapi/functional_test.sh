curl -X POST \
  -H 'Content-Type: application/json' \
  -d '{"name": "rock"}' \
  http://localhost:8000/items

curl "http://localhost:8000/items?name=rock"

curl -X PUT \
  -H 'Content-Type: application/json' \
  -d '{"name": "rock", "quantity": 100}' \
  http://localhost:8000/items

curl -X DELETE \
  -H 'Content-Type: application/json' \
  -d '{"name": "rock"}' \
  http://localhost:8000/items

curl "http://localhost:8000/items?name=rock"