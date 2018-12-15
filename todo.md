Order of operations:
1. import summary json and:
    1. extract:
        - gw to date map
        - player id's
        - id to name map
        - name to id map
    2. generate:
        - date to gw map
        

# other
- [ ] add canonical list of team names to check its a team we've seen in the DB before and have a mapping for
- [ ] look into the proportion of their team's goals/assists each player is getting and multiply by expected number of goals for the team to get an expected number of goals and assists for that player.
```python
player_xG, player_xA = team_xG * [player_prop_team_G, player_prop_team_A]
player_xFPL = (
    (player_xG * pts_G) + 
    (player_xA * pts_A) + 
    (player_xYellow * pts_Yellow) + 
    (player_xRed * pts_Red) +
)
```
