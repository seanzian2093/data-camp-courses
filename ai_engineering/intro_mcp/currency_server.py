import requests
import sqlite3
from mcp.server.mcpserver import MCPServer

# Create an MCP server instance
mcp = MCPServer("Currency Converter")

# Connect to the database on startup
conn = sqlite3.connect("currencies.db")
conn.row_factory = sqlite3.Row

# Adding typing to the function arguments and return object
# @mcp.tool()
# def _convert_currency(amount, from_currency, to_currency):
#     """
#     Convert an amount from one currency to another using current exchange rates.

#     Args:
#         amount: The amount to convert
#         from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
#         to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')

#     Returns:
#         A string with the conversion result and exchange rate
#     """
#     # API endpoint for Frankfurter - we don't have access to it
#     url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"

#     # 1. Make the API request
#     response = requests.get(url)

#     # 2. Extract the currency exchange rate from the response
#     data = response.json()
#     rate = data['rates'].get(to_currency)

#     if rate is None:
#         return f"Could not find exchange rate for {from_currency} to {to_currency}"

#     # 3. Calculate the converted amount
#     converted_amount = amount * rate
#     return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"
      
# @mcp.tool()
# def convert_currency_robust(amount: float, from_currency: str, to_currency: str) -> str:
#     """
#     Convert an amount from one currency to another using current exchange rates.

#     Args:
#         amount: The amount to convert
#         from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
#         to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')

#     Returns:
#         A string with the conversion result and exchange rate
#     """
#     url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"
#     # Implement try-except to gracefully handle errors
#     try:
#         # Add a 10-second timeout so the request does not hang
#         r = requests.get(url, timeout=10)
#         r.raise_for_status()
#         data = r.json()
#         rate = data["rates"].get(to_currency)
#         if rate is None:
#             return f"Could not find exchange rate for {from_currency} to {to_currency}"
#         return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency} (Rate: {rate})"
#     except requests.exceptions.RequestException as e:
#         return f"Error converting currency: {e}"

# print(convert_currency(10, "USD", "EUR"))

@mcp.tool()
def convert_currency(amount, from_currency, to_currency):
    """
    Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')

    Returns:
        A string with the conversion result and exchange rate
    """

    # Fake API response
    rates = {
        "USD": {"EUR": 0.91, "GBP": 0.78, "USD": 1.0},
        "EUR": {"USD": 1.1, "GBP": 0.85, "EUR": 1.0},
        "GBP": {"USD": 1.28, "EUR": 1.18, "GBP": 1.0}
    }
    rate = rates.get(from_currency, 0.0).get(to_currency, 0.0)  # Example fixed exchange rate
    converted_amount = amount * rate
    return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"

# Define a resource for the currencies file
@mcp.resource("file://currencies.txt")
def get_currencies() -> str:
    """
    Get the list of currency names published by the European Central Bank for currency conversion.

    Returns:
        Contents of the currencies.txt file with currency names
    """
    # Open currencies.txt and read the data
    try:
        with open('currencies.txt', 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "currencies.txt file not found"

# Define a prompt for currency conversion
@mcp.prompt(title="Currency Conversion")
def convert_currency_prompt(currency_request: str) -> str:
    return f"""You are a currency conversion assistant.

Your task is to:
1. Extract the amount and source currency from the user's natural language input.
2. Identify the target currency.
3. Use the conversion tool to convert the amount.

Rules:
- If the amount or currencies are ambiguous or missing, ask the user for clarification.
- Use only supported currency codes (e.g., USD, EUR, GBP).

User's currency conversion request: {currency_request}"""

# Add lookup_currencies(prefix): find rows where name or code contains prefix
@mcp.tool()
def lookup_currencies(prefix: str) -> str:
    """Find currencies whose code or name contains the given prefix."""
    try:
        # Use parameterized query and LIMIT 50
        cursor = conn.execute(
            "SELECT code, name FROM currencies WHERE name LIKE ? OR code LIKE ? LIMIT 50",
            (f"%{prefix}%", f"%{prefix}%"),
        )
        rows = cursor.fetchall()
        return "\n".join(f"{row['code']} - {row['name']}" for row in rows)
    except sqlite3.Error as e:
        return f"Database error: {e}"

# print(lookup_currencies("Euro"))

# Create an MCP resource of database
@mcp.resource("db://currencies")
def get_currencies() -> str:
    """
    Get the list of currency names published by the European Central Bank for currency conversion.

    Returns:
        One line per currency (code - name), from the database.
    """
    try:
        # Query the database
        cursor = conn.execute("SELECT code, name FROM currencies")
        rows = cursor.fetchall()
        return "\n".join(f"{row['code']} - {row['name']}" for row in rows)
    except sqlite3.Error as e: return f"Error: {e}"
# result = get_currencies()
# print(result[:200] + "..." if len(result) > 200 else result)

# Add lookup_currencies(prefix): find rows where name or code contains prefix from a db
@mcp.tool()
def lookup_currencies(prefix: str) -> str:
    """Find currencies whose code or name contains the given prefix."""
    try:
        # Use parameterized query and LIMIT 50
        cursor = conn.execute(
            "SELECT code, name FROM currencies WHERE name LIKE ? OR code LIKE ? LIMIT 50",
            (f"%{prefix}%", f"%{prefix}%"),
        )
        rows = cursor.fetchall()
        return "\n".join(f"{row['code']} - {row['name']}" for row in rows)
    except sqlite3.Error as e:
        return f"Database error: {e}"

# print(lookup_currencies("Euro"))

if __name__ == "__main__":
    mcp.run()