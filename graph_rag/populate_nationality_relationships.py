"""Populate nationality relationships for the fake author data."""

import ast
from pathlib import Path

from helpers import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME


POPULATE_FAKE_DATA_PATH = Path(__file__).with_name("populate_fake_data.py")


AUTHOR_NATIONALITIES = {
    "William Shakespeare": "England",
    "Jane Austen": "England",
    "Charles Dickens": "England",
    "Leo Tolstoy": "Russia",
    "Fyodor Dostoevsky": "Russia",
    "Mark Twain": "United States",
    "Ernest Hemingway": "United States",
    "George Orwell": "England",
    "F. Scott Fitzgerald": "United States",
    "Virginia Woolf": "England",
    "James Joyce": "Ireland",
    "Franz Kafka": "Austria-Hungary",
    "Gabriel Garcia Marquez": "Colombia",
    "J.R.R. Tolkien": "England",
    "C.S. Lewis": "Ireland",
    "Agatha Christie": "England",
    "Arthur Conan Doyle": "Scotland",
    "Herman Melville": "United States",
    "Nathaniel Hawthorne": "United States",
    "Charlotte Bronte": "England",
    "Emily Bronte": "England",
    "Homer": "Greece",
    "Miguel de Cervantes": "Spain",
    "Victor Hugo": "France",
    "Alexandre Dumas": "France",
    "Oscar Wilde": "Ireland",
    "Mary Shelley": "England",
    "Bram Stoker": "Ireland",
    "Toni Morrison": "United States",
    "Harper Lee": "United States",
    "J.D. Salinger": "United States",
    "Kurt Vonnegut": "United States",
    "Aldous Huxley": "England",
    "Ray Bradbury": "United States",
    "Isaac Asimov": "Russia",
    "Ursula K. Le Guin": "United States",
    "Philip K. Dick": "United States",
    "Jack Kerouac": "United States",
    "John Steinbeck": "United States",
    "William Faulkner": "United States",
    "Dante Alighieri": "Italy",
    "Gustave Flaubert": "France",
    "Marcel Proust": "France",
    "Albert Camus": "France",
    "Jean-Paul Sartre": "France",
    "Chinua Achebe": "Nigeria",
    "George Eliot": "England",
    "Thomas Hardy": "England",
}


def load_fake_authors() -> dict[str, list[str]]:
    module = ast.parse(POPULATE_FAKE_DATA_PATH.read_text())
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "AUTHORS_AND_BOOKS":
                    return ast.literal_eval(statement.value)
    raise ValueError("Could not find AUTHORS_AND_BOOKS in populate_fake_data.py")


def build_people() -> list[dict[str, str]]:
    authors_and_books = load_fake_authors()
    missing_authors = set(authors_and_books) - set(AUTHOR_NATIONALITIES)
    extra_authors = set(AUTHOR_NATIONALITIES) - set(authors_and_books)

    if missing_authors or extra_authors:
        raise ValueError(
            "Nationality mapping does not match fake author data: "
            f"missing={sorted(missing_authors)}, extra={sorted(extra_authors)}"
        )

    return [
        {"name": name, "nation": AUTHOR_NATIONALITIES[name]}
        for name in authors_and_books
    ]


if __name__ == "__main__":
    from langchain_neo4j import Neo4jGraph

    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    people = build_people()

    cypher = """
    UNWIND $people AS person
    MERGE (p:Person {name: person.name})
    MERGE (n:Nationality {nation: person.nation})
    MERGE (p)-[:HAS_NATIONALITY]->(n)
    RETURN count(p) AS relationships_processed
    """

    result = graph.query(cypher, {"people": people})
    relationships_processed = result[0]["relationships_processed"] if result else 0
    print(f"Processed {relationships_processed} nationality relationships.")

    graph.refresh_schema()
    print(graph.schema)