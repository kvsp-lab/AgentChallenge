def format_currency(amount, currency="BRL"):
    return f"{currency} {amount:,.2f}"

def get_status_color(status):
    colors = {
        "High Confidence": "green",
        "Medium Confidence": "orange",
        "Low Confidence": "red",
        "Needs Review": "red",
        "Confirmed by User": "blue",
        "Corrected by User": "purple"
    }
    return colors.get(status, "gray")
