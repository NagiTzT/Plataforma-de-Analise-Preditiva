import json
import math
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, Mock

import free_football_data as free

CSV = ('Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H\n'
       'E0,01/09/2026,Manchester United,Liverpool,1,0,H,0,,0,,9.99\n'
       'E0,02/09/2026,Manchester United,Liverpool,,,D,1,2,1,2,1.99\n'
       'E0,03/09/2026,Manchester United,Liverpool,0,1,H,1,2,1,2,2.99\n')
NOW = int(datetime(2026,9,11,tzinfo=timezone.utc).timestamp())


class FreeSourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = str(Path(self.folder.name)/'source.db')
        with closing(free.connect(self.db)) as conn:
            conn.executemany('INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                             free.parse_csv(CSV,'https://football-data.co.uk/test',NOW-100))
            conn.commit()

    def tearDown(self):
        self.folder.cleanup()

    def test_no_odds_in_records_missing_is_not_zero(self):
        records = free.parse_csv(CSV,'source',NOW)
        self.assertEqual(len(records),1)
        metrics = json.loads(records[0][9])
        self.assertEqual(metrics['home_shots'],0)
        self.assertIsNone(metrics['away_shots'])
        self.assertNotIn('9.99',str(records))
        self.assertEqual(set(metrics),{f'{side}_{metric}' for side in ('home','away') for metric in free.METRICS})

    def test_identity_rejects_other_categories(self):
        source = free.FreeStatistics(self.db)
        self.assertIsNotNone(source.match_teams('Manchester Utd','Liverpool FC','England'))
        for suffix in (' U21',' Women',' II',' B'):
            self.assertIsNone(source.match_teams('Manchester United'+suffix,'Liverpool','England'))
        self.assertIsNone(source.match_teams('Manchester United','Liverpool','Scotland'))

    def test_cutoff_capture_and_metric_coverage(self):
        source = free.FreeStatistics(self.db)
        identity = source.match_teams('Manchester United','Liverpool')
        self.assertEqual(source.features(*identity,cutoff=NOW-101)['free_home_all_games'],0)
        online = source.features(*identity,cutoff=NOW)
        self.assertEqual(online['free_home_all_games'],1)
        self.assertEqual(online['free_home_all_shots_for'],0)
        self.assertTrue(math.isnan(online['free_home_all_shots_against']))
        # Same/current fixture cannot be its own history, even in reconstruction.
        start = datetime(2026,9,1,tzinfo=timezone.utc).timestamp()
        self.assertEqual(source.features(*identity,cutoff=start,historical=True)['free_home_all_games'],0)
        self.assertEqual(source.features(*identity,cutoff=start+4*86400,historical=True)['free_home_all_games'],0)

    def test_index_block_persists_backoff(self):
        response = Mock(status_code=403,content=b'forbidden')
        with patch.object(free.time,'time',return_value=NOW), patch.object(free.requests,'get',return_value=response) as get:
            self.assertEqual(free.collect_bulk(self.db)['failures'][0]['status'],403)
            self.assertEqual(free.collect_bulk(self.db),{'provider_backoff':1})
            get.assert_called_once()

    def test_fresh_cache_needs_zero_requests(self):
        url = 'https://football-data.co.uk/mmz4281/2627/data.zip'
        with closing(free.connect(self.db)) as conn:
            conn.execute('INSERT INTO bulk_index VALUES (?,?,?)',(free.INDEX,NOW,json.dumps([url])))
            conn.execute('INSERT INTO sources VALUES (?,?,?,?)',(url,NOW,200,'hash'))
            conn.commit()
        with patch.object(free.time,'time',return_value=NOW), patch.object(free.requests,'get') as get:
            self.assertEqual(free.collect_bulk(self.db)['requests'],0)
            get.assert_not_called()

    def test_enrichment_freezes_and_reuses_on_failure(self):
        main = str(Path(self.folder.name)/'main.db')
        games = [{'ID':'new','Timestamp':NOW+86400,'Time Casa':'Manchester United','Time Fora':'Liverpool','Liga':'England'},
                 {'ID':'started','Timestamp':NOW-1,'Time Casa':'Manchester United','Time Fora':'Liverpool','Liga':'England'}]
        with patch.object(free.time,'time',return_value=NOW), patch.object(free,'collect_bulk',side_effect=free.requests.Timeout):
            self.assertEqual(free.enrich_games(main,games,self.db)['linked_games'],1)
        with patch.object(free.time,'time',return_value=NOW+2):
            features = free.load_pregame_features(main,'new',NOW+86400)
            self.assertEqual(features['free_home_all_games'],1)
            self.assertNotIn('free_home_all_shots_against',features)
            self.assertEqual(free.load_pregame_features(main,'new',NOW-1),{})
        with closing(sqlite3.connect(main)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM free_pregame_snapshots').fetchone()[0],1)


if __name__ == '__main__':
    unittest.main()
