import json
import sqlite3
import unittest

from pregame_snapshot_store import (CORE,init_archive,read_recent_snapshots,save_recent_snapshot,
                                    snapshot_asof,snapshot_index)
from sofascore_intelligence import _recent_event_rows, _live_recent_features


class RecentSnapshotStoreTests(unittest.TestCase):
    def setUp(self):
        self.conn=sqlite3.connect(':memory:')
        self.conn.execute('''CREATE TABLE pregame_recent_form_snapshots (
            match_id TEXT,side TEXT,provider TEXT,provider_team_id TEXT,captured_at INTEGER,
            cutoff_timestamp INTEGER,freshest_match_timestamp INTEGER,games_json TEXT,
            features_json TEXT,coverage REAL,PRIMARY KEY(match_id,side))''')

    def tearDown(self):
        self.conn.close()

    def values(self,captured=30000,score=1,cutoff=100000):
        return ('target','home','sofascore','h',captured,cutoff,1000,
                json.dumps([{'match_id':'past','team_id':'h','goals_for':score}]),'{}',1.)

    def test_read_legacy_database_does_not_create_archive(self):
        self.conn.execute('INSERT INTO pregame_recent_form_snapshots VALUES(?,?,?,?,?,?,?,?,?,?)',self.values())
        self.assertEqual(len(read_recent_snapshots(self.conn)),1)
        self.assertIsNone(self.conn.execute("SELECT name FROM sqlite_master WHERE name='pregame_recent_form_archive'").fetchone())

    def test_refresh_keeps_original_evidence_and_capture_time(self):
        init_archive(self.conn)
        save_recent_snapshot(self.conn,self.values())
        save_recent_snapshot(self.conn,self.values(50000,2))
        all_rows=read_recent_snapshots(self.conn)
        self.assertEqual(len(all_rows),2)
        index=snapshot_index(all_rows)
        target=dict(match_id='target',captured_at=31000,kickoff=100000)
        original=snapshot_asof(index,target,'home')
        self.assertEqual(original['captured_at'],30000)
        self.assertEqual(json.loads(original['games_json'])[0]['goals_for'],1)
        self.assertEqual(self.conn.execute('SELECT captured_at FROM pregame_recent_form_snapshots').fetchone()[0],50000)

    def test_repeated_save_does_not_duplicate_same_evidence(self):
        init_archive(self.conn)
        for _ in range(3):
            save_recent_snapshot(self.conn,self.values())
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM pregame_recent_form_archive').fetchone()[0],1)

    def test_existing_legacy_record_is_archived_before_replacement(self):
        self.conn.execute('INSERT INTO pregame_recent_form_snapshots VALUES(?,?,?,?,?,?,?,?,?,?)',self.values())
        init_archive(self.conn)
        save_recent_snapshot(self.conn,self.values(50000,2))
        self.assertEqual({r['captured_at'] for r in read_recent_snapshots(self.conn)},{30000,50000})

    def test_conflicting_same_second_is_kept_but_cannot_be_selected(self):
        init_archive(self.conn)
        save_recent_snapshot(self.conn,self.values())
        save_recent_snapshot(self.conn,self.values(score=2))
        index=snapshot_index(read_recent_snapshots(self.conn))
        self.assertIsNone(snapshot_asof(index,dict(match_id='target',captured_at=31000,kickoff=100000),'home'))
        self.assertEqual(len(read_recent_snapshots(self.conn)),2)

    def test_postgame_refresh_never_replaces_pregame_for_replay(self):
        init_archive(self.conn)
        save_recent_snapshot(self.conn,self.values())
        save_recent_snapshot(self.conn,self.values(110000,5))
        target=dict(match_id='target',captured_at=31000,kickoff=100000)
        snapshot=snapshot_asof(snapshot_index(read_recent_snapshots(self.conn)),target,'home')
        self.assertEqual(snapshot['captured_at'],30000)
        target['kickoff']=100001
        self.assertIsNone(snapshot_asof(snapshot_index(read_recent_snapshots(self.conn)),target,'home'))


class LiveRecentIdentityTests(unittest.TestCase):
    def event(self,mid='1',home='h',goals=1):
        return dict(id=mid,startTimestamp=1000,status={'type':'finished'},
            homeTeam={'id':home,'name':'Same Team'},awayTeam={'id':'a','name':'Opponent'},
            homeScore={'normaltime':goals},awayScore={'normaltime':0})

    def test_live_intake_collapses_aliases_before_ten_game_limit(self):
        rows=_recent_event_rows({'events':[self.event(),self.event('2')]},'h','Same Team',100000)
        self.assertEqual(len(rows),1)

    def test_live_intake_refuses_conflicting_score_aliases(self):
        rows=_recent_event_rows({'events':[self.event(),self.event('2',goals=4)]},'h','Same Team',100000)
        self.assertEqual(rows,[])

    def test_similar_name_does_not_override_known_provider_id(self):
        self.assertEqual(_recent_event_rows({'events':[self.event(home='wrong')]},'h','Same Team',100000),[])

    def test_bad_provider_timestamps_do_not_abort_the_radar(self):
        for value in (None,True,'invalid',[],float('nan'),1.5):
            event=self.event()
            event['startTimestamp']=value
            self.assertEqual(_recent_event_rows({'events':[event]},'h','Same Team',100000),[])

    def test_new_quality_features_do_not_claim_five_recent_games_when_missing(self):
        # Empty histories need no DB lookups and retain the incumbent schema.
        features=_live_recent_features('unused','home',[],100000)
        self.assertEqual(features['live_recent_home_games'],0)
        self.assertEqual(features['form_fresh180_home_available'],0)


if __name__=='__main__':
    unittest.main()
