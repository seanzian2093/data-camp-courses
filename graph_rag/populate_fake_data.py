"""Populate the Neo4j graph with ~200 Book nodes written by well-known authors."""
import re
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from helpers import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME

# Well-known authors mapped to a handful of their famous works
AUTHORS_AND_BOOKS = {
    "William Shakespeare": ["Hamlet", "Macbeth", "Othello", "King Lear"],
    "Jane Austen": ["Pride and Prejudice", "Sense and Sensibility", "Emma", "Persuasion"],
    "Charles Dickens": ["A Tale of Two Cities", "Great Expectations", "Oliver Twist", "David Copperfield"],
    "Leo Tolstoy": ["War and Peace", "Anna Karenina", "The Death of Ivan Ilyich", "Resurrection"],
    "Fyodor Dostoevsky": ["Crime and Punishment", "The Brothers Karamazov", "The Idiot", "Demons"],
    "Mark Twain": ["The Adventures of Tom Sawyer", "Adventures of Huckleberry Finn", "The Prince and the Pauper", "A Connecticut Yankee in King Arthur's Court"],
    "Ernest Hemingway": ["The Old Man and the Sea", "A Farewell to Arms", "For Whom the Bell Tolls", "The Sun Also Rises"],
    "George Orwell": ["1984", "Animal Farm", "Homage to Catalonia", "Down and Out in Paris and London"],
    "F. Scott Fitzgerald": ["The Great Gatsby", "Tender Is the Night", "This Side of Paradise", "The Beautiful and Damned"],
    "Virginia Woolf": ["Mrs Dalloway", "To the Lighthouse", "Orlando", "The Waves"],
    "James Joyce": ["Ulysses", "Dubliners", "A Portrait of the Artist as a Young Man", "Finnegans Wake"],
    "Franz Kafka": ["The Trial", "The Metamorphosis", "The Castle", "Amerika"],
    "Gabriel Garcia Marquez": ["One Hundred Years of Solitude", "Love in the Time of Cholera", "Chronicle of a Death Foretold", "Autumn of the Patriarch"],
    "J.R.R. Tolkien": ["The Hobbit", "The Fellowship of the Ring", "The Two Towers", "The Return of the King"],
    "C.S. Lewis": ["The Lion, the Witch and the Wardrobe", "Prince Caspian", "The Voyage of the Dawn Treader", "The Magician's Nephew"],
    "Agatha Christie": ["Murder on the Orient Express", "And Then There Were None", "The Murder of Roger Ackroyd", "Death on the Nile"],
    "Arthur Conan Doyle": ["A Study in Scarlet", "The Hound of the Baskervilles", "The Sign of Four", "The Adventures of Sherlock Holmes"],
    "Herman Melville": ["Moby-Dick", "Bartleby, the Scrivener", "Typee", "Billy Budd"],
    "Nathaniel Hawthorne": ["The Scarlet Letter", "The House of the Seven Gables", "The Blithedale Romance", "Twice-Told Tales"],
    "Charlotte Bronte": ["Jane Eyre", "Shirley", "Villette", "The Professor"],
    "Emily Bronte": ["Wuthering Heights"],
    "Homer": ["The Iliad", "The Odyssey"],
    "Miguel de Cervantes": ["Don Quixote"],
    "Victor Hugo": ["Les Miserables", "The Hunchback of Notre-Dame", "Ninety-Three", "The Man Who Laughs"],
    "Alexandre Dumas": ["The Three Musketeers", "The Count of Monte Cristo", "Twenty Years After", "The Man in the Iron Mask"],
    "Oscar Wilde": ["The Picture of Dorian Gray", "The Importance of Being Earnest", "Lady Windermere's Fan", "De Profundis"],
    "Mary Shelley": ["Frankenstein"],
    "Bram Stoker": ["Dracula"],
    "Toni Morrison": ["Beloved", "Song of Solomon", "The Bluest Eye", "Sula"],
    "Harper Lee": ["To Kill a Mockingbird", "Go Set a Watchman"],
    "J.D. Salinger": ["The Catcher in the Rye", "Franny and Zooey", "Nine Stories", "Raise High the Roof Beam, Carpenters"],
    "Kurt Vonnegut": ["Slaughterhouse-Five", "Cat's Cradle", "Breakfast of Champions", "Mother Night"],
    "Aldous Huxley": ["Brave New World", "Island", "Point Counter Point", "Eyeless in Gaza"],
    "Ray Bradbury": ["Fahrenheit 451", "The Martian Chronicles", "Something Wicked This Way Comes", "Dandelion Wine"],
    "Isaac Asimov": ["Foundation", "I, Robot", "The Caves of Steel", "The Gods Themselves"],
    "Ursula K. Le Guin": ["The Left Hand of Darkness", "The Dispossessed", "A Wizard of Earthsea", "The Tombs of Atuan"],
    "Philip K. Dick": ["Do Androids Dream of Electric Sheep?", "The Man in the High Castle", "Ubik", "A Scanner Darkly"],
    "Jack Kerouac": ["On the Road", "The Dharma Bums", "Big Sur", "Visions of Cody"],
    "John Steinbeck": ["The Grapes of Wrath", "Of Mice and Men", "East of Eden", "Cannery Row"],
    "William Faulkner": ["The Sound and the Fury", "As I Lay Dying", "Absalom, Absalom!", "Light in August"],
    "Dante Alighieri": ["Inferno", "Purgatorio", "Paradiso"],
    "Gustave Flaubert": ["Madame Bovary", "Sentimental Education", "Bouvard et Pecuchet", "Three Tales"],
    "Marcel Proust": ["Swann's Way", "Within a Budding Grove", "The Guermantes Way", "Time Regained"],
    "Albert Camus": ["The Stranger", "The Plague", "The Fall", "The Myth of Sisyphus"],
    "Jean-Paul Sartre": ["Nausea", "No Exit", "The Age of Reason", "Being and Nothingness"],
    "Chinua Achebe": ["Things Fall Apart", "No Longer at Ease", "Arrow of God", "A Man of the People"],
    "George Eliot": ["Middlemarch", "Silas Marner", "The Mill on the Floss", "Adam Bede"],
    "Thomas Hardy": ["Tess of the d'Urbervilles", "Far from the Madding Crowd", "The Mayor of Casterbridge", "Jude the Obscure"],
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def build_graph_document() -> GraphDocument:
    nodes = []
    relationships = []
    book_count = 0

    for author, titles in AUTHORS_AND_BOOKS.items():
        author_node = Node(type="Person", id=slugify(author), properties={"name": author})
        nodes.append(author_node)

        for title in titles:
            if book_count >= 200:
                break
            book_node = Node(type="Book", id=slugify(title), properties={"title": title})
            nodes.append(book_node)
            relationships.append(Relationship(source=book_node, target=author_node, type="WRITTEN_BY"))
            book_count += 1

    return GraphDocument(nodes=nodes, relationships=relationships)


if __name__ == "__main__":
    graph_document = build_graph_document()
    print(f"Prepared {sum(1 for n in graph_document.nodes if n.type == 'Book')} Book nodes "
          f"and {sum(1 for n in graph_document.nodes if n.type == 'Person')} Person nodes.")

    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    graph.add_graph_documents([graph_document])

    graph.refresh_schema()
    print(graph.schema)
