import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()



''' PLACEHOLDER VALUES FOR INPUT AND OUTPUT WHILE APPLICATION IS BEING WORKED ON '''



ph_df = pd.read_csv("https://drive.google.com/uc?export=download&id=1QUQwDWzbuGizJ9rVxzB-H7uio3eE6uRa", sep='\t')
sub_names = ph_df['subreddit_title'].tolist()

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: str = Field(..., example='This is an example title')
    x2: str = Field(..., example='this is an example selftext or link')
    

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)
    y_pred = random.choice(sub_names)
    df1 = ph_df[ph_df['subreddit_title'] == y_pred]
    sub_des = df1['subreddit_description'].iloc[0]
    return {
        'prediction': y_pred,
        'description': sub_des
    }
