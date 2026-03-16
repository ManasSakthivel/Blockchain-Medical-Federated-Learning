#!/usr/bin/env python3
"""
Fund Ganache Account Script
Sends ETH from Ganache's first account to the app's account to resolve insufficient funds error.
"""
from web3 import Web3

# Ganache RPC URL
GANACHE_URL = 'http://127.0.0.1:7545'
# Funded Ganache account (index 0, fixed mnemonic)
SENDER_ADDRESS = '0x627306090abaB3A6e1400e9345bC60c78a8BEf57'
SENDER_PRIVATE_KEY = '0x4f3edf983ac636a65a842ce7c78d9aa706d3b113b37e74e3a7e61a8634850e19'
# App's account (from your private_keys.json)
RECEIVER_ADDRESS = '0x30B6940714B7c8318E5Cd02aD743a0032CA898B5'
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

# Minimum balance threshold (in ETH)
MIN_BALANCE_ETH = 5
# Amount to top up (in ETH)
TOP_UP_AMOUNT_ETH = 10

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), 'Web3 is not connected to Ganache!'

receiver_balance = w3.from_wei(w3.eth.get_balance(RECEIVER_ADDRESS), 'ether')
print(f"Current balance of {RECEIVER_ADDRESS}: {receiver_balance} ETH")

if receiver_balance < MIN_BALANCE_ETH:
    nonce = w3.eth.get_transaction_count(SENDER_ADDRESS)
    tx = {
        'nonce': nonce,
        'to': RECEIVER_ADDRESS,
        'value': w3.to_wei(TOP_UP_AMOUNT_ETH, 'ether'),
        'gas': 21000,
        'gasPrice': w3.to_wei('2', 'gwei'),
    }
    signed_tx = w3.eth.account.sign_transaction(tx, SENDER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"Topped up {TOP_UP_AMOUNT_ETH} ETH from {SENDER_ADDRESS} to {RECEIVER_ADDRESS}")
    print(f"Transaction hash: {w3.to_hex(tx_hash)}")
else:
    print(f"No top-up needed. Balance is sufficient.")
