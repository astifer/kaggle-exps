import torch
from torch import nn, Tensor

class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int, drop_p: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.do1 = nn.Dropout1d(p=drop_p)
        self.do2 = nn.Dropout1d(p=drop_p)
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.LeakyReLU()
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.do1(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.act(x)

        return x


class MyModel(nn.Module):
    def __init__(self, hidden_size: int, drop_p: float):
        super().__init__()
        self.emb_gender = nn.Embedding(2, embedding_dim=16)
        self.emb_customer_type = nn.Embedding(2, embedding_dim=16)
        self.emb_travel_type = nn.Embedding(2, embedding_dim=16)
        self.emb_class = nn.Embedding(3, embedding_dim=16)
        self.emb_time_convenience = nn.Embedding(6, embedding_dim=16)
        self.emb_online_book = nn.Embedding(6, embedding_dim=16)
        self.emb_checkin_service = nn.Embedding(6, embedding_dim=16)
        self.emb_online_boarding = nn.Embedding(6, embedding_dim=16)
        self.emb_gate_location = nn.Embedding(6, embedding_dim=16)
        self.emb_onboard_service = nn.Embedding(6, embedding_dim=16)
        self.emb_gate_location = nn.Embedding(6, embedding_dim=16)
        self.emb_seat_comfort = nn.Embedding(6, embedding_dim=16)
        self.emb_leg_room_service = nn.Embedding(6, embedding_dim=16)
        self.emb_cleanliness = nn.Embedding(6, embedding_dim=16)
        self.emb_food_drink = nn.Embedding(6, embedding_dim=16)
        self.emb_inflight = nn.Embedding(6, embedding_dim=16)
        self.emb_inflight_wifi = nn.Embedding(6, embedding_dim=16)
        self.emb_inflight_entertainment = nn.Embedding(6, embedding_dim=16)
        self.emb_baggage_handle = nn.Embedding(6, embedding_dim=16)

        self.numeric_linear = nn.Linear(4, 16)
        self.numeric_concat = nn.Linear(304, hidden_size)

        self.block1 = BaseBlock(hidden_size, drop_p)
        self.block2 = BaseBlock(hidden_size, drop_p)
        self.block3 = BaseBlock(hidden_size, drop_p)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        
        x_gender = self.emb_gender(cat_features['Gender'])
        x_customer_type = self.emb_customer_type(cat_features['Customer Type'])
        x_travel_type = self.emb_travel_type(cat_features['Type of Travel'])
        x_class = self.emb_class(cat_features['Class'])
        x_time_convenience = self.emb_time_convenience(cat_features['Departure and Arrival Time Convenience'])
        x_online_book = self.emb_online_book(cat_features['Ease of Online Booking'])
        x_checkin_service = self.emb_checkin_service(cat_features['Check-in Service'])
        x_online_boarding = self.emb_online_boarding(cat_features['Online Boarding'])
        x_gate_location = self.emb_gate_location(cat_features['Gate Location'])
        x_onboard_service = self.emb_onboard_service(cat_features['On-board Service'])
        x_seat_comfort = self.emb_seat_comfort(cat_features['Seat Comfort'])
        x_leg_room_service = self.emb_leg_room_service(cat_features['Leg Room Service'])
        x_cleanliness = self.emb_cleanliness(cat_features['Cleanliness'])
        x_food_drink = self.emb_food_drink(cat_features['Food and Drink'])
        x_inflight = self.emb_inflight(cat_features['In-flight Service'])
        x_inflight_wifi = self.emb_inflight_wifi(cat_features['In-flight Wifi Service'])
        x_inflight_entertainment = self.emb_inflight_entertainment(cat_features['In-flight Entertainment'])
        x_baggage_handle = self.emb_baggage_handle(cat_features['Baggage Handling'])


        stacked_numeric = torch.stack([ 
            numeric_features['Age'], 
            numeric_features['Flight Distance'], 
            numeric_features['Departure Delay'],
            numeric_features['Arrival Delay'],
            ], dim=-1)
        
        x_numeric = self.numeric_linear(stacked_numeric)

        x_emb_staked = torch.cat([
            x_numeric,
            x_gender,
            x_customer_type,
            x_travel_type,
            x_class,
            x_time_convenience,
            x_online_book,
            x_checkin_service,
            x_online_boarding,
            x_gate_location,
            x_onboard_service,
            x_seat_comfort,
            x_leg_room_service,
            x_cleanliness,
            x_food_drink,
            x_inflight,
            x_inflight_wifi,
            x_inflight_entertainment,
            x_baggage_handle
        ], dim=-1)

        x_total = self.numeric_concat(x_emb_staked)

        x_total = self.block1(x_total) + x_total
        x_total = self.block2(x_total) + x_total
        x_total = self.block3(x_total) + x_total

        result = self.linear_out(x_total)

        result = torch.sigmoid(result.view(-1))

        return result