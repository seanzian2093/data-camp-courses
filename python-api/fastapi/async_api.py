from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    # description: str


# A pretend database
items = {"rock", "paper", "scissors"}


app = FastAPI()


@app.delete("/items")
async def delete_item(item: Item):
    name = item.name
    if name not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    items.remove(name)
    return {"message": "Item Deleted"}
