curl -X DELETE \
  -H 'Content-Type: application/json' \
  -d '{"name": "rock"}' \
  http://localhost:8000/items

curl -X DELETE \
  -H 'Content-Type: application/json' \
  -d '{"name": "roll"}' \
  http://localhost:8000/items