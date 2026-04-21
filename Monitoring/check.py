import pandas as pd
df = pd.read_csv("../mi_cartera/historial_operaciones.csv")
compras = df[df['Accion'].str.strip().str.upper() == 'COMPRA']
ventas = df[df['Accion'].str.strip().str.upper() == 'VENTA']

# We started with a capital of €10M. Let's calculate the total capital deployed, total revenue from sales, and net profit/loss.
# We buy at "Precio_Ejecutado" and sell at "Precio_Ejecutado", that includes the "CT" (costs per share).

print("Total Capital Deployed in Purchases:", (compras['Cantidad'].abs() * compras['Precio_Ejecutado']).sum())
print("Total Revenue from Sales:", (ventas['Cantidad'].abs() * ventas['Precio_Ejecutado']).sum())
net_profit_loss = (ventas['Cantidad'].abs() * ventas['Precio_Ejecutado']).sum() - (compras['Cantidad'].abs() * compras['Precio_Ejecutado']).sum()
print("Net Profit/Loss:", net_profit_loss)


# let's calculate the costs associated with the transactions, such as commissions and taxes. Column 'CT' represents the costs per share, so we can calculate the total costs for both purchases and sales.
print("Total Costs for Purchases:", (compras['Cantidad'].abs() * compras['CT']).sum())
print("Total Costs for Sales:", (ventas['Cantidad'].abs() * ventas['CT']).sum())

# We want to see if we are leverage and how much, this is check if we have negative cash balance right now.
# Calculate the cash balance after all transactions
cash_balance = 10000000 - (compras['Cantidad'].abs() * compras['Precio_Ejecutado']).sum() + (ventas['Cantidad'].abs() * ventas['Precio_Ejecutado']).sum() - (compras['Cantidad'].abs() * compras['CT']).sum() + (ventas['Cantidad'].abs() * ventas['CT']).sum()
print("Current Cash Balance:", cash_balance)
