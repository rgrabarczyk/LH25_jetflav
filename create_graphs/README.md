A script that reads in JSONs made by code in create_jsons and creates DGL graphs that can be used as inputs to LundNet.
makeLundJSONDGL.py builds the Lund Trees and specifies the variables at each nodes, so one should look in there for any potential modifications.
Node that if one chooses to change the number of dimensions at each node, an appropriate change must be made in multilund.py.
To create the graphs simply do 
    python3 makeLundJSONDGL.py < jsonpath > -nev 400
  