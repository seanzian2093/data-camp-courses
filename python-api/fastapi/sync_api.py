from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str


# A pretend database
items = {"bananas": "Yellow fruit"}


app = FastAPI()


@app.get("/")
def root(name: str = "Alan"):
    return {"message": f"Hello {name}"}


@app.post("/")
def root(item: Item):
    name = item.name
    return {"message": f"Hello {name}"}


@app.put("/items")
def update_item(item: Item):
    name = item.name
    items[name] = item.description
    return item


@app.delete("/items")
def delete_item(item: Item):
    name = item.name
    try:
        deleted_item = items.pop(name)
    except HTTPException as e:
        raise HTTPException(status_code=404, detail="Item not found")
    return deleted_item


# run the app by executing the following command
# fastapi dev main1.py

# curl \
#   -H 'Content-Type: application/json' \
#   "http://localhost:8000?name=Steve"

# curl -X POST \
#   -H 'Content-Type: application/json' \
#   -d '{"name": "bananas"}' \
#   http://localhost:8000


# curl -X PUT \
#   -H 'Content-Type: application/json' \
#   -d '{"name": "bananas", "description": "Delicious!"}' \
#   http://localhost:8000/items

# curl -X DELETE \
#   -H 'Content-Type: application/json' \
#   -d '{"name": "bananas", "description": "Delicious!"}' \
#   http://localhost:8000/items
