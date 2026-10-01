import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from phase2_observations import append
from phase4_metadata import competition_metadata, country_from_league, repair_frozen_metadata
from phase4_context_collection import collect_context, candidate_pool, _safe_history, claim_daily_network_collection
from phase4_readiness import summarize
from phase4_research import capture_eligibility_candidate


METRIC_NAMES = ('xg','shots_on_target','big_chances')
STATS = {'statistics':[{'period':'ALL','groups':[{'statisticsItems':[
    {'key':k,'homeValue':0,'awayValue':0}
    for k in ('expectedGoals','shotsOnGoal','bigChanceCreated')]}]}]}


class CaptureRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/'main.db'
        self.now=time.time();self.kickoff=self.now+12*3600
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('CREATE TABLE pregame_recent_form_snapshots(match_id TEXT,side TEXT,provider TEXT,captured_at REAL,cutoff_timestamp REAL,games_json TEXT)')
            columns=','.join(f'{m}_{d} REAL' for m in METRIC_NAMES for d in ('for','against'))
            conn.execute('CREATE TABLE sofascore_team_match_profiles(match_id TEXT,team_name TEXT,captured_at REAL,metric_presence_json TEXT,'+columns+')')

    def tearDown(self):
        self.temp.cleanup()

    def game(self,mid='target',run='run',odds=(2.2,3.2,2.4),league='Brazil Serie A'):
        capture_eligibility_candidate(self.db,run,mid,list(odds),self.kickoff,self.now,
            provider='test',home_team='Alpha',away_team='Beta',league=league)
        for stage in ('collection_started','collection_finished','phase4_run_health_v1','phase4_shadow_health_v1'):
            append(self.db,run,'',stage,{'status':'PASS'})
        with closing(sqlite3.connect(self.db.with_name('phase2_observations.db'))) as conn:
            sha=conn.execute("SELECT sha256 FROM observations WHERE run_id=? AND match_id=? AND stage='phase4_prefilter_v1'",(run,mid)).fetchone()[0]
        return {'ID':mid,'Timestamp':self.kickoff,'Time Casa':'Alpha','Time Fora':'Beta',
            'Liga':league,'Research_Run_ID':run,'Eligibility_Group':'A_ELIGIBLE' if odds[0]>1.99 else 'B1_HOME_LOW','Parent_SHA256':sha}

    def history(self,name,side):
        return [{'match_id':f'{side}-{i}','team_name':name,'team_id':side,
            'opponent_name':'Other','opponent_id':'other','start_timestamp':self.now-(i+2)*86400,
            'is_home':True,'goals_for':0,'goals_against':0} for i in range(3)]

    def profiles(self,game):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            for side,name in [('home','Alpha'),('away','Beta')]:
                history=self.history(name,side)
                conn.execute('INSERT INTO pregame_recent_form_snapshots VALUES (?,?,?,?,?,?)',
                    (game['ID'],side,'allsports',self.now,self.kickoff,json.dumps(history)))
                for row in history:
                    keys=[f'{m}_{d}' for m in METRIC_NAMES for d in ('for','against')]
                    conn.execute('INSERT INTO sofascore_team_match_profiles VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (row['match_id'],name,self.now,json.dumps(keys),0,0,0,0,0,0))

    def test_country_scalar_nested_category_and_frozen_prefix(self):
        self.assertEqual(competition_metadata({'championship':{'country':'Brazil'}})['country'],'Brazil')
        event={'tournament':{'uniqueTournament':{'category':{'country':{'name':'Japan'}}}}}
        self.assertEqual(competition_metadata(event)['country'],'Japan')
        self.assertEqual(country_from_league('Brazil Serie A'),'Brazil')
        self.assertEqual(country_from_league('Bosnia & Herzegovina 1st League'),'Bosnia and Herzegovina')
        self.assertEqual(country_from_league('UAE Cup'),'United Arab Emirates')
        self.assertEqual(country_from_league('Europe Friendlies'),'')
        self.assertEqual(country_from_league('United League'),'')

    def test_repair_preserves_original_hash_and_is_idempotent(self):
        self.game()
        sidecar=self.db.with_name('phase2_observations.db')
        with closing(sqlite3.connect(sidecar)) as conn:
            before=conn.execute("SELECT sha256,payload FROM observations WHERE stage='phase4_prefilter_v1'").fetchone()
        self.assertEqual(repair_frozen_metadata(self.db)['repaired'],1)
        self.assertEqual(repair_frozen_metadata(self.db)['repaired'],1)
        with closing(sqlite3.connect(sidecar)) as conn:
            after=conn.execute("SELECT sha256,payload FROM observations WHERE stage='phase4_prefilter_v1'").fetchone()
            count=conn.execute("SELECT COUNT(*) FROM observations WHERE stage='phase4_metadata_repair_v1'").fetchone()[0]
        self.assertEqual(before,after);self.assertEqual(count,1)
        report=summarize(self.db,sidecar)
        self.assertEqual(report['representativeness']['country_repairs_from_original_league'],1)
        self.assertEqual(report['coverage_by_country_season_provider']['country'],{'Brazil':1})

    def test_zero_measured_metrics_are_valid_and_main_db_is_unchanged(self):
        game=self.game();self.profiles(game)
        before=hashlib.sha256(self.db.read_bytes()).hexdigest()
        request=Mock(side_effect=AssertionError('cache should suffice'))
        result=collect_context(self.db,games=[game],api_get=request,max_requests=6)
        self.assertEqual(result['snapshots_saved'],1)
        self.assertEqual(result['xg_available'],1)
        request.assert_not_called()
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(),before)
        report=summarize(self.db,self.db.with_name('phase2_observations.db'))
        self.assertEqual(report['coverage_dashboard']['family_available_counts']['xg'],1)
        self.assertEqual(report['coverage_dashboard']['separate_covariate_snapshots'],1)
        self.assertFalse(report['outcomes_opened'])

    def test_pool_excludes_failed_runs_and_finished_fixtures(self):
        self.game('ok');self.game('failed','failed')
        append(self.db,'failed','','phase4_pipeline_error_v1',{'error_type':'test'})
        self.assertEqual([g['ID'] for g in candidate_pool(self.db,time.time())],['ok'])
        self.assertEqual(candidate_pool(self.db,self.kickoff+1),[])

    def test_history_excludes_current_future_and_wrong_identity(self):
        good=self.history('Alpha','home')[0]
        rows=[good,{**good,'match_id':'target'},
            {**good,'match_id':'live','start_timestamp':self.now-3600},
            {**good,'match_id':'other','team_name':'Alpha Women'}]
        self.assertEqual([r['match_id'] for r in _safe_history(rows,'Alpha','target',self.now)],[good['match_id']])

    def test_completion_budget_requests_only_previous_statistics(self):
        game=self.game();calls=[]
        def get(url):
            calls.append(url)
            if url.endswith('/match/target'):
                return {'event':{'id':'target','startTimestamp':int(self.kickoff),
                    'homeTeam':{'name':'Alpha','id':'home'},'awayTeam':{'name':'Beta','id':'away'}}}
            if '/matches/previous/' in url:
                side='home' if '/team/home/' in url else 'away';name='Alpha' if side=='home' else 'Beta'
                events=[{'id':r['match_id'],'startTimestamp':int(r['start_timestamp']),
                    'status':{'type':'finished'},'homeTeam':{'name':name,'id':side},
                    'awayTeam':{'name':'Other','id':'other'},'homeScore':{'normaltime':0},
                    'awayScore':{'normaltime':0}} for r in self.history(name,side)]
                return {'events':events}
            if url.endswith('/statistics'):return STATS
            raise AssertionError(url)
        result=collect_context(self.db,games=[game],api_get=get,max_requests=9,max_games=1)
        self.assertEqual(result['snapshots_saved'],1)
        self.assertLessEqual(len(calls),9)
        self.assertNotIn('/api/match/target/statistics',' '.join(calls))
        self.assertEqual(sum(url.endswith('/statistics') for url in calls),6)

    def test_after_kickoff_never_requests_or_freezes_context(self):
        game=self.game();get=Mock()
        result=collect_context(self.db,games=[game],api_get=get,clock=lambda:self.kickoff+1)
        self.assertEqual(result['already_started'],1);get.assert_not_called()
        self.assertEqual(result['snapshots_saved'],0)

    def test_daily_network_claim_is_durable_even_without_finish(self):
        self.game()
        run,claimed=claim_daily_network_collection(self.db,self.now,60,12)
        self.assertTrue(claimed)
        again,claimed=claim_daily_network_collection(self.db,self.now+1,60,12)
        self.assertFalse(claimed);self.assertEqual(run,again)

    def test_auxiliary_coverage_cannot_unlock_original_holdout(self):
        game=self.game();self.profiles(game)
        collect_context(self.db,games=[game],max_requests=0)
        report=summarize(self.db,self.db.with_name('phase2_observations.db'))
        self.assertEqual(report['coverage_dashboard']['family_available_counts']['xg'],1)
        self.assertEqual(report['evaluation_trigger']['checks']['family_observations']['current_min'],0)
        self.assertFalse(report['coverage_dashboard']['separate_covariates_unlock_original_holdout'])

    def test_report_recognizes_measured_free_shots_without_inventing_xg(self):
        game=self.game()
        f={f'free_{s}_all_target_{d}_games':3 for s in ('home','away') for d in ('for','against')}
        append(self.db,'run','target','prediction_phase4_prefilter_v1',{'features':f,'data_coverage':{'families':{}}})
        report=summarize(self.db,self.db.with_name('phase2_observations.db'))
        self.assertEqual(report['coverage_dashboard']['family_available_counts']['shots_on_target'],1)
        self.assertEqual(report['coverage_dashboard']['family_available_counts'].get('xg',0),0)

    def test_report_rejects_covariate_with_wrong_parent(self):
        self.game()
        append(self.db,'run','target','phase4_covariates_v2',{'parent_sha256':'wrong',
            'captured_at':time.time(),'separate_from_champion_inputs':True})
        report=summarize(self.db,self.db.with_name('phase2_observations.db'))
        self.assertEqual(report['coverage_dashboard']['rejected_covariate_snapshots'],1)
        self.assertEqual(report['coverage_dashboard']['separate_covariate_snapshots'],0)


class HttpBudgetTests(unittest.TestCase):
    def test_key_rotation_cannot_exceed_research_request_budget(self):
        import robo_auto as bot
        response=Mock(status_code=429,headers={'x-ratelimit-requests-remaining':'0'},text='quota')
        with patch.object(bot,'_cached_api_response',return_value=None), \
             patch.object(bot,'_reserve_rapidapi_key',return_value=('test-key','test-id',0)) as reserve, \
             patch.object(bot,'_sync_rapidapi_response'), \
             patch.object(bot,'RAPIDAPI_MIN_INTERVAL_SECONDS',0), \
             patch.object(bot.requests,'get',return_value=response) as request:
            self.assertIsNone(bot.safe_api_get('https://example.invalid/unit-test',max_http_requests=1))
        self.assertEqual(request.call_count,1);self.assertEqual(reserve.call_count,1)


if __name__=='__main__':unittest.main()
