def compute_totals(records, tax_rate, discount):
    subtotal = 0
    item_count = 0
    for record in records:
        if record.status == "paid":
            subtotal += record.amount
            item_count += record.quantity
        elif record.status == "refunded":
            subtotal -= record.amount
    taxed = subtotal * (1 + tax_rate)
    final = taxed - (taxed * discount)
    return {"total": final, "items": item_count}


def invoice_label(invoice):
    return "invoice-" + str(invoice.id)
