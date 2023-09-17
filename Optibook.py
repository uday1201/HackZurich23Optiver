


# imports
import datetime as dt
import time
from time import sleep
import random
import logging
import sys

# connect to exchange
from optibook.synchronous_client import Exchange
exchange = Exchange()
exchange.connect()
logging.getLogger('client').setLevel('ERROR')

# set constants
MAX_POSITIONS = 200
DELTA = 0.1
GAMMA = 0.1
SLEEP_TIME = 0.1
VOLUME = 5

# get stock names
LIQUID_STOCK_ID = "PHILIPS_A"
ILLIQUID_STOCK_ID = "PHILIPS_B"

def initialise_illiquid():
    """
    Find entry prices onto the illiquid market
    
    Args:
        None
        
    Returns:
        LIQUID_BID (float) : entry bid for liquid market
        LIQUID_ASK (float) : entry ask for liquid market
    
    """
    initial_found = False
    while not initial_found:
        illiquid_book = exchange.get_last_price_book(ILLIQUID_STOCK_ID)
        illiquid_book_bids = illiquid_book.bids
        illiquid_book_asks = illiquid_book.asks
        try:
            illiquid_bid = illiquid_book_bids[0].price
            illiquid_ask = illiquid_book_asks[0].price
            initial_found = True
        except IndexError: 
            pass
        
    spread = abs(illiquid_ask - illiquid_bid)
        
    return illiquid_bid + 0.1*spread, illiquid_ask - 0.1*spread
    
illiquid_bid, illiquid_ask = initialise_illiquid()

def trading_loop(t):
    """
    Main trading loop
    
    Args:
        t (float) : current time
    """
    
    
    # get stock books
    liquid_book = exchange.get_last_price_book(LIQUID_STOCK_ID)
    illiquid_book = exchange.get_last_price_book(ILLIQUID_STOCK_ID)
    
    # get bids and asks for both books
    liquid_book_bids = liquid_book.bids
    illiquid_book_bids = illiquid_book.bids
    liquid_book_asks = liquid_book.asks
    illiquid_book_asks = illiquid_book.asks
    
    # search for best bid and ask price in books
    try:
        liquid_best_bid_price = liquid_book_bids[0].price
        illiquid_best_bid_price = illiquid_book_bids[0].price
        liquid_best_ask_price = liquid_book_asks[0].price
        illiquid_best_ask_price = illiquid_book_asks[0].price
    except IndexError:
        return
    
    # --------------------------------------------------------------------------------------- #
    # Adjust bid and ask prices on illiquid market every 10 ticks
    # And make sure that spread on illiquid market is wider than liquid market
    # --------------------------------------------------------------------------------------- #
    
    # every 10 ticks adjust spreads to keep them competative
    global illiquid_bid, illiquid_ask
    updated_bid = False
    updated_ask = False
    if(t % 10 == 0):
        illiquid_bid += DELTA
        illiquid_ask -= DELTA
        updated_bid = True
        updated_ask = True
        
    # ensure spead is tighter on illiquid market
    if(illiquid_best_ask_price - illiquid_ask < GAMMA):
        illiquid_ask = illiquid_best_ask_price - GAMMA
    if(illiquid_bid - illiquid_best_bid_price < GAMMA):
        illiquid_bid = illiquid_best_bid_price + GAMMA
        
    
    # --------------------------------------------------------------------------------------- #
    # Place orders on the liquid market
    # Use limit orders to ensure we have time
    # --------------------------------------------------------------------------------------- #
    
    # get current positions
    liquid_position = exchange.get_positions()[LIQUID_STOCK_ID]
    illiquid_position = exchange.get_positions()[ILLIQUID_STOCK_ID]
    
    # short
    if(illiquid_position > -MAX_POSITIONS and updated_ask):
        
        # delete any current orders
        exchange.delete_orders(ILLIQUID_STOCK_ID)
        
        # place new ask
        exchange.insert_order(ILLIQUID_STOCK_ID, price = illiquid_ask, volume = VOLUME, side='ask', order_type='limit')
        
    # long
    if(illiquid_position < MAX_POSITIONS and updated_bid):
        
        # delete any current orders
        exchange.delete_orders(ILLIQUID_STOCK_ID)
        
        # place new bid
        exchange.insert_order(ILLIQUID_STOCK_ID, price = illiquid_bid, volume = VOLUME, side = 'bid', order_type = 'limit')
        
    
    # --------------------------------------------------------------------------------------- #
    # If we hold positions, attempt to hedge on liquid market
    # use ioc orders
    # --------------------------------------------------------------------------------------- #

    total_position = liquid_position + illiquid_position
    if(total_position > 0):
        exchange.insert_order(LIQUID_STOCK_ID, price = liquid_best_bid_price, volume = abs(total_position), side='ask', order_type='ioc')
    elif(total_position < 0):
        exchange.insert_order(LIQUID_STOCK_ID, price = liquid_best_ask_price, volume = abs(total_position), side = 'bid', order_type = 'ioc')
        
    # --------------------------------------------------------------------------------------- #
    # Print results for each run
    # --------------------------------------------------------------------------------------- #
        
    if t%10 == 0:
        print(t)
        positions = exchange.get_positions()
        pnl = exchange.get_pnl()
        print(f'\nPositions after: {positions}')
        print(f'\nPnL after: {pnl:.2f}')
        
        
    # add delay to add time for limit order to go through
    sleep(SLEEP_TIME)
    
    # --------------------------------------------------------------------------------------- #
    # Check for outstanding orders
    # If all orders go through we can increase spread
    # --------------------------------------------------------------------------------------- #
    
    # check for remaining orders from limit order on illiquid market
    illiquid_outstanding = exchange.get_outstanding_orders(ILLIQUID_STOCK_ID)
    outstanding_illiquid_bid = False
    outstanding_illiquid_ask = False
    for outstanding in illiquid_outstanding.values():
        if(outstanding.side == "bid"): outstanding_illiquid_bid = True
        if(outstanding.side == 'ask'): outstanding_liquid_bid = True
        
    # if no outstanding orders, increase spread
    if(outstanding_illiquid_bid == False): illiquid_bid -= 2*DELTA
    if(outstanding_illiquid_ask == False): illiquid_ask += 2*DELTA
    
    # --------------------------------------------------------------------------------------- #
    # Sell positions if we can profit
    # --------------------------------------------------------------------------------------- #
    
    # get stock books
    liquid_book = exchange.get_last_price_book(LIQUID_STOCK_ID)
    illiquid_book = exchange.get_last_price_book(ILLIQUID_STOCK_ID)
    
    # get bids and asks for both books
    liquid_book_bids = liquid_book.bids
    illiquid_book_bids = illiquid_book.bids
    liquid_book_asks = liquid_book.asks
    illiquid_book_asks = illiquid_book.asks
        
    # search for best bid and ask in books
    try:
        # get best bid and ask price
        liquid_best_bid_price = liquid_book_bids[0].price
        liquid_best_ask_price = liquid_book_asks[0].price
        illiquid_curr_bid_price = illiquid_book_bids[0].price
        illiquid_curr_ask_price = illiquid_book_asks[0].price
    
    except IndexError:
        return
    
    # need to buy liquid
    if(liquid_position < 0 and illiquid_position > 0):
        counter = 0
        
        # price to buy liquid (liquid ask) needs to be lower than price to sell illiqud (illiquid bid)
        while(illiquid_curr_bid_price >= liquid_book_asks[counter].price and counter < len(liquid_book_asks)):
            illiquid_position = exchange.get_positions()[ILLIQUID_STOCK_ID]
            liquid_position = exchange.get_positions()[LIQUID_STOCK_ID]
            unwinding_vol = min(abs(illiquid_book_bids[0].volume), abs(liquid_book_asks[counter].volume), abs(liquid_position), abs(illiquid_position))
            if(unwinding_vol <= 0): break
            exchange.insert_order(ILLIQUID_STOCK_ID, price = float(illiquid_curr_bid_price), side = 'ask', volume = unwinding_vol, order_type = 'ioc')
            exchange.insert_order(LIQUID_STOCK_ID, price = float(liquid_book_asks[counter].price), side = 'bid', volume = unwinding_vol, order_type = 'ioc')
        
            counter += 1
            
    # need to buy in illiquid
    if(liquid_position > 0 and illiquid_position < 0):
        counter = 0
        
        # price to buy illiquid (ask illiquuid) needs to be lower than price to sell liquid (bid liquid)
        while(illiquid_curr_ask_price <= liquid_book_bids[counter].price and counter < len(liquid_book_bids)):
            illiquid_position = exchange.get_positions()[ILLIQUID_STOCK_ID]
            liquid_position = exchange.get_positions()[LIQUID_STOCK_ID]
            unwinding_vol = min(abs(illiquid_book_asks[0].volume), abs(liquid_book_bids[counter].volume), abs(liquid_position), abs(illiquid_position))
            if(unwinding_vol <= 0): break
            exchange.insert_order(ILLIQUID_STOCK_ID, price = float(illiquid_curr_ask_price), side = 'bid', volume = unwinding_vol, order_type = 'ioc')
            exchange.insert_order(LIQUID_STOCK_ID, price = float(liquid_book_bids[counter].price), side = 'ask', volume = unwinding_vol, order_type = 'ioc')
        
            counter += 1





def main():
    t = 0
    while True:
        trading_loop(t)
        t += 1
        
# run code
if __name__ == "__main__":
    main()