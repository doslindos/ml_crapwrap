select
   t.id,
   t.spotify_id as data,
   t.name,
   max(p.popularity) as label
from 
   dexmusic_track t
   join dexmusic_trackpopularity p
     on (p.track_id = t.id)
where 
   p.popularity > 0
group by
   t.id	 
;
