import datetime as dt
import time
import random
import logging
from optibook.synchronous_client import Exchange
#from libs import print_positions_and_pnl, round_down_to_tick, round_up_to_tick
from IPython.display import clear_output
import predict

logging.getLogger('client').setLevel('ERROR')   


from math import floor, ceil

def print_positions_and_pnl(exchange):
    positions = exchange.get_positions()
    pnl = exchange.get_pnl()

    print('Positions:')
    for instrument_id in positions:
        print(f'  {instrument_id:10s}: {positions[instrument_id]:4.0f}')

    print(f'\nPnL: {pnl:.2f}')

    
def round_down_to_tick(price, tick_size):
    """
    Rounds a price down to the nearest tick, e.g. if the tick size is 0.10, a price of 0.97 will get rounded to 0.90.
    """
    return floor(price / tick_size) * tick_size


def round_up_to_tick(price, tick_size):
    """
    Rounds a price up to the nearest tick, e.g. if the tick size is 0.10, a price of 1.34 will get rounded to 1.40.
    """
    return ceil(price / tick_size) * tick_size    



class Trader:
    PRICE_RETREAT_PER_LOT = 0.04
    POSITION_LIMIT = 100
    QUOTED_VOLUME = 10
    
    exchange = {}
    # the avg spread as a percentage
    avgSpread = 0
    # min spread threshhold in percent 
    minSpread = 0.05
    # 
    spreads = []
    
    def __init__(self, exchange):
        self.exchange = exchange
        
    def get_info(self,instrument):
        trade_ticks = self.exchange.poll_new_trade_ticks(instrument)
        trade_history = self.exchange.get_trade_history(instrument)
        outstanding_orders = self.exchange.get_outstanding_orders(instrument)
        last_price_book = self.exchange.get_last_price_book(instrument)
        positions = self.exchange.get_positions()
        
        # show fetched information
        print("trade_ticks ", trade_ticks)
        print("trade_history ", trade_history)
        print("outstanding_orders ", outstanding_orders)
        print("last_price_book ", last_price_book)
        print("positions ", positions)
        
        return trade_ticks, trade_history, outstanding_orders, last_price_book, positions

    # The Rrice we think the INSTRUMENT should have, based on that we gonna calculate our other variables
    def calculate_theoretical_price(self,instrument_order_book, position, bias = 0):
        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        mid_price = (best_bid_price + best_ask_price) / 2.0 
        
        # here we gonna insert the bias on what we think the price should be, based on:
        # - sentiment_bias from the tweets, should it go up or not 
        # TODO Implement the bias, should be pluged in
        
        # ('NVDA', 0.04147530719637871)
        #expected_movement = predict.predict(tweet[0].post)
        #if tweet = exchange.poll_new_social_media_feeds()
#        print(tweet)
        
        # Calculate our fair/theoretical price based on the market mid price and our current position
        # current implementation
        theoretical_price = mid_price - self.PRICE_RETREAT_PER_LOT * position
        
        return theoretical_price

    # this is the amount of credit we expect on a quote we insert into the market
    def calculate_min_expected_credit_symetric(self,instrument,theoretical_price):
        # this should be scaled on the current instrument price and not fixed 
        # this is variable and could be imporved
        scaler = 0.01
        expected_credit = theoretical_price * scaler
        return expected_credit

    # it is the delta of the current best price on both sides, with the 
    def calculate_bids(self,instrument,theoretical_price):
        symetric_credit_delta = self.calculate_min_expected_credit_symetric(instrument,theoretical_price)
        
        # TODO improve, 
        
        # Calculate final bid and ask prices to insert
        bid_price = round_down_to_tick(theoretical_price - symetric_credit_delta, instrument.tick_size)
        ask_price = round_up_to_tick(theoretical_price + symetric_credit_delta, instrument.tick_size)
        
        return bid_price,ask_price

    # this is calculating the new or "should be" exposure to a certain INSTRUMENT
    # - this is based on the volumes in the order book and shozld not be constant
    def calculate_quoted_volume_old(self,position,price_book):
        max_volume_to_buy = self.POSITION_LIMIT - position
        max_volume_to_sell = self.POSITION_LIMIT + position

        # the volumes should take into account 
        # - how the order book looks
        # if the average ask price is close to the mid price, expected bid, ask_price ?
        # TODO

        bid_volume = min(self.QUOTED_VOLUME, max_volume_to_buy)
        ask_volume = min(self.QUOTED_VOLUME, max_volume_to_sell)
        
        return bid_volume, ask_volume
    
    def calculate_quoted_volume(self, position, price_book):
        max_volume_to_buy = self.POSITION_LIMIT - position
        max_volume_to_sell = self.POSITION_LIMIT + position

        if price_book.bids and price_book.asks:
            # Calculate mid price
            mid_price = (price_book.bids[0].price + price_book.asks[0].price) / 2

            # Calculate market depth and imbalance
            total_bid_volume = sum(bid.volume for bid in price_book.bids)
            total_ask_volume = sum(ask.volume for ask in price_book.asks)
            market_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            # Dynamic QUOTED_VOLUME based on market conditions
            base_quoted_volume = min(total_bid_volume, total_ask_volume) / 2
            QUOTED_VOLUME = base_quoted_volume * (1 - abs(market_imbalance))

            # Adjust volumes based on market conditions and position limits
            max_volume_to_buy = min(max_volume_to_buy, QUOTED_VOLUME * (1 + market_imbalance))
            max_volume_to_sell = min(max_volume_to_sell, QUOTED_VOLUME * (1 - market_imbalance))

        else:
            # If there are no bids or asks, set QUOTED_VOLUME to a default value (e.g., 10)
            QUOTED_VOLUME = 10
            max_volume_to_buy = min(max_volume_to_buy, QUOTED_VOLUME)
            max_volume_to_sell = min(max_volume_to_sell, QUOTED_VOLUME)

        return max_volume_to_buy, max_volume_to_sell
    
    # new
    def edit_quotes(self,instrument, bid_price, ask_price, bid_volume, ask_volume, outstanding_instrument_orders):
        # TODO make the function handle more than one outstanding order per side 
        def get_first_value(dictionary):                                                                                                           
            return next(iter(dictionary.values())) 
        outstanding_asks = self.filter_orders_by_side(outstanding_instrument_orders, "ask")
        outstanding_bids = self.filter_orders_by_side(outstanding_instrument_orders, "bids")
        print("outstanding_bids",outstanding_bids)
        print("outstanding_asks",outstanding_asks)
        ##### ASKS #####
        
        
        
        def insert_bid():
            if bid_volume > 0 and (bid_volume + ask_volume) < 100:
                print("bid_volume", bid_volume)
                print("ask_volume", bid_volume)
                # Insert new bid limit order on the market
                self.exchange.insert_order(
                    instrument_id=instrument.instrument_id,
                    price=bid_price,
                    volume=int(bid_volume) + 1,
                    side='bid',
                    order_type='limit',
                )
                
                # Wait for some time to avoid breaching the exchange frequency limit
                time.sleep(0.05)
                
        def insert_ask():
            if ask_volume > 0 and (bid_volume + ask_volume) < 100:
                print("ask_volume", ask_volume)
                print("bid_volume", bid_volume)
                # Insert new ask limit order on the market
                self.exchange.insert_order(
                    instrument_id=instrument.instrument_id,
                    price=ask_price,
                    volume=int(ask_volume) + 1,
                    side='ask',
                    order_type='limit',
                )

                # Wait for some time to avoid breaching the exchange frequency limit
                time.sleep(0.05)
                # add some test 
        
        if len(outstanding_asks) > 0:
            # edit it, take the open order, which should be just one in this example and edit it to our desired volume 
            outstanding_ask = get_first_value(outstanding_asks)
            
            if outstanding_ask.price == ask_price:
                # same price, jsut change the exposure
                print(instrument.instrument_id,outstanding_ask.order_id, ask_volume)
                test = int(outstanding_ask.order_id)
                print(test)
                test2 = outstanding_ask.order_id
                self.exchange.amend_order(instrument.instrument_id, order_id=outstanding_ask.order_id, volume=ask_volume)
            else:
                self.exchange.delete_orders(instrument.instrument_id)
                #self.exchange.amend_order(instrument.instrument_id,order_id=outstanding_ask.order_id, volume=0)#outstanding_ask.order_id, 0)
                insert_ask()
                insert_bid()
        else:
            insert_ask()
            
            
        ##### BIDS #####
                
        if len(outstanding_bids) > 0:
            # edit it, take the open order, which should be just one in this example and edit it to our desired volume 
            outstanding_bid = get_first_value(outstanding_bids)
            
            if outstanding_bid.price == bid_price:
                # same price, jsut change the exposure
                self.exchange.amend_order(instrument.instrument_id,order_id=outstanding_bid.order_id, volume=bid_volume)#outstanding_bid.order_id,bid_volume)
            else: # reduce the volume to 0 and set an order at a different place
                self.exchange.delete_orders(instrument.instrument_id)
                #self.exchange.amend_order(instrument.instrument_id,order_id=outstanding_bid.order_id, volume=0)#outstanding_bid.order_id, 0)
                insert_bid()
                insert_ask()
        else:
            insert_bid()

    # old 
    def insert_quotes(self, instrument, bid_price, ask_price, bid_volume, ask_volume):

        if bid_volume > 0:
            # Insert new bid limit order on the market
            self.exchange.insert_order(
                instrument_id=instrument.instrument_id,
                price=bid_price,
                volume=bid_volume,
                side='bid',
                order_type='limit',
            )
            
            # Wait for some time to avoid breaching the exchange frequency limit
            time.sleep(0.05)

        if ask_volume > 0:
            # Insert new ask limit order on the market
            self.exchange.insert_order(
                instrument_id=instrument.instrument_id,
                price=ask_price,
                volume=ask_volume,
                side='ask',
                order_type='limit',
            )

            # Wait for some time to avoid breaching the exchange frequency limit
            time.sleep(0.05)
            # add some test comment
            
    # calculate volatilry based on the prevSpread and the current spread and avgSpread
    def isVolatile(self):
        # diff of these 2
        # - volume ? 
        isVolatilityIncreasing = (self.curSpread - self.avgSpread) < 0
        return isVolatilityIncreasing

    def spread(self,bid_price, ask_price,best_bid_price,best_ask_price):
        # Spread & Liquidity
        bid_ask_spread = best_ask_price - best_bid_price 
        spread_percentage = (bid_ask_spread / best_ask_price) * 100
        print(f'Spread Info: {bid_ask_spread:>6.2f}, {spread_percentage:>6.2f}')
        
        return bid_ask_spread,spread_percentage
    
    # more sophisticated version
    
    def calculate_volatility(orders):                                                                                                                          
        prices = np.array([order.price for order in orders])                                                                                                   
        volumes = np.array([order.volume for order in orders])                                                                                                 
        returns = np.diff(prices) / prices[:-1]                                                                                                                
        volume_weighted_returns = returns * volumes[1:]                                                                                                        
        volatility = np.std(volume_weighted_returns)                                                                                                           
        return volatility    
   
    def calculate_liquidity(bid_prices, ask_prices, volatility, depth):                                                                                        
        spreads = np.array([order.price for order in ask_prices]) - np.array([order.price for order in bid_prices])                                            
        average_spread = np.mean(spreads)                                                                                                                      
        adjusted_spread = average_spread / volatility / depth                                                                                                  
        max_possible_spread = np.max([np.max([order.price for order in bid_prices]), np.max([order.price for order in ask_prices])]) - np.min([np.min([order.price for order in bid_prices]), np.min([order.price for order in ask_prices])])                                                     
        liquidity = adjusted_spread / max_possible_spread                                                                                                      
        return liquidity   
   
   
    def calculate_depth(bid_volumes, ask_volumes):                                                                                                             
        total_bid_volume = np.sum([order.volume for order in bid_volumes])                                                                                     
        total_ask_volume = np.sum([order.volume for order in ask_volumes])                                                                                     
        depth = total_bid_volume + total_ask_volume                                                                                                            
        return depth                                                                                                                                           
                         
  
    def is_liquid(liquidity, threshold=0.001):                                                                                                                 
        return liquidity > threshold  
    
    
    def first_layer(self,instrument,position,instrument_order_book,best_bid_price,best_ask_price,bias = 0):    
        # Calculate our fair/theoretical price based on the market mid price and our current position
        theoretical_price = self.calculate_theoretical_price(instrument_order_book, position)
        
        bid_price,ask_price = self.calculate_bids(instrument, theoretical_price)
        # Calculate bid and ask volumes to insert, taking into account the exchange position_limit
        bid_volume,ask_volume = self.calculate_quoted_volume(position,instrument_order_book)

        # Display information for tracking the algorithm's actions
        
        self.printResults(bid_price,ask_price,bid_volume,ask_volume,instrument)
            
        return bid_price, ask_price, bid_volume, ask_volume
    
    def second_layer():
        # TODO
        print("ass")
    
    def trade(self, instrument, position, outstanding_orders,instrument_order_book, bias):
        # calc best bid and ask based just for printeing 
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        
        # first layer for fast response 
        bid_price, ask_price, bid_volume, ask_volume = self.first_layer(instrument,position,instrument_order_book,best_bid_price,best_ask_price,bias)

        # save spread development
        bid_ask_spread,spread_percentage = self.spread(bid_price,ask_price,best_bid_price,best_ask_price)
        self.curSpread = spread_percentage
        self.spreads.append(self.curSpread)
        self.avgSpread = sum(self.spreads) / len(self.spreads)

        # 
        self.edit_quotes(instrument, bid_price, ask_price, bid_volume/2, ask_volume/2,outstanding_orders)
    
    def filter_orders_by_instrument(self,orders, instrument_id):                                                                                                    
        return {order_id: order for order_id, order in orders.items() if order.instrument_id == instrument_id} 
    def filter_orders_by_instrument_(self,orders, instrument_id):                                                                                                    
        return [order for order in orders if order.instrument_id == instrument_id]
    
                                                                                                                                                                
    def filter_orders_by_side(self,orders, side):                                                                                                                   
        return {order_id: order for order_id, order in orders.items() if order.side == side}                                                                   
                                                                                                                                                             


    def filter_orders_by_side_(self,orders, side):                                                                                                                   
        return [order for order in orders if order.side == side]  
        
    def run(self):
        self.exchange.connect()
        
        while True:
            # TODO do request management
            self.printWhileHeader()
            
            # get info from the exchange 

            
            
            for instrument in self.exchange.get_instruments().values():
                
                trade_ticks, trade_history, outstanding_orders, instrument_order_book, positions = self.get_info(instrument.instrument_id)
                
                if not (instrument_order_book and instrument_order_book.bids and instrument_order_book.asks):
                    print(f'{instrument.instrument_id:>6s} --     INCOMPLETE ORDER BOOK')
                    continue
                position = positions[instrument.instrument_id]
                
                outstanding_instrument_orders = self.filter_orders_by_instrument(outstanding_orders, instrument.instrument_id)
                
                # last sentiment for a instrument
                #tweet = exchange.poll_new_social_media_feeds()
                #bias = predict.predict(tweet[0].post)
                bias = 0
                
                
                self.trade(instrument, position,outstanding_instrument_orders,instrument_order_book, bias)
                
            # Wait for a few moments to refresh the quotes
            # TODO request priotisation
            time.sleep(0.05)
            
            # Clear the displayed information after waiting
            clear_output(wait=True)
    
        
    def printWhileHeader(self):
        print(f'')
        print(f'-----------------------------------------------------------------')
        print(f'TRADE LOOP ITERATION ENTERED AT {str(dt.datetime.now()):18s} UTC.')
        print(f'-----------------------------------------------------------------')
        # Display our own current positions in all stocks, and our PnL so far
        print_positions_and_pnl(self.exchange)
        print(f'')
        print(f'          (ourbid) mktbid :: mktask (ourask)')
        
    def printResults(self,bid_price,ask_price,best_ask_price,best_bid_price,instrument):
        print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')
        
           
###### LETS GO

exchange = Exchange()

trader = Trader(exchange)

trader.run()

