import copy
import unittest

from recent_history_integrity import clean_recent_history, fresh_sequence_features


NOW = 1800000000


def event(mid='1',opponent='a',age=10,goals=0,own='h',home=1):
    return dict(match_id=mid,team_id=own,opponent_id=opponent,
                start_timestamp=NOW-age*86400,is_home=home,
                goals_for=goals,goals_against=0)


class RecentHistoryIntegrityTests(unittest.TestCase):
    def test_alias_ids_are_one_fixture_not_two_performances(self):
        rows=[event(),event('2')]
        clean,meta=clean_recent_history(rows,NOW,team_id='h')
        self.assertEqual(len(clean),1)
        self.assertEqual(meta['duplicate_rows'],1)
        self.assertEqual(rows,[event(),event('2')])

    def test_same_fixture_with_conflicting_scores_is_quarantined(self):
        rows=[event(),event('2',goals=3)]
        clean,meta=clean_recent_history(rows,NOW)
        self.assertEqual(clean,[])
        self.assertEqual(meta['conflicting_rows'],2)

    def test_event_id_conflict_propagates_to_its_other_alias(self):
        rows=[event('1'),event('1',age=20),event('2')]
        self.assertEqual(clean_recent_history(rows,NOW)[0],[])

    def test_distinct_dates_opponents_and_venues_are_not_merged(self):
        rows=[event(),event('2',age=11),event('3',opponent='b'),event('4',home=0)]
        self.assertEqual(len(clean_recent_history(rows,NOW)[0]),4)

    def test_age_limit_is_opt_in_and_never_fills_with_ancient_matches(self):
        rows=[event(str(i),age=10+i*100) for i in range(10)]
        self.assertEqual(len(clean_recent_history(rows,NOW)[0]),10)
        clean,meta=clean_recent_history(rows,NOW,max_age_days=180)
        self.assertEqual(len(clean),2)
        self.assertEqual(meta['stale_rows_excluded'],8)
        self.assertEqual(meta['five_games_available'],0)

    def test_missing_scores_are_unknown_and_measured_zero_is_valid(self):
        self.assertEqual(len(clean_recent_history([event()],NOW)[0]),1)
        for invalid in (None,True,-1,float('nan'),.5):
            self.assertEqual(clean_recent_history([event(goals=invalid)],NOW)[0],[])

    def test_wrong_identity_and_target_or_unfinished_event_are_rejected(self):
        self.assertEqual(clean_recent_history([event()],NOW,team_id='other')[0],[])
        self.assertEqual(clean_recent_history([event()],NOW,target_id='1')[0],[])
        self.assertEqual(clean_recent_history([event(age=0)],NOW)[0],[])

    def test_missing_ids_are_not_joined_by_names(self):
        rows=[event(),event('2')]
        for row in rows:
            row.pop('opponent_id'); row['opponent_name']='Same Name'
        self.assertEqual(len(clean_recent_history(rows,NOW)[0]),2)

    def test_future_conflict_cannot_quarantine_earlier_known_result(self):
        rows=[event(),event('1',age=-1,goals=5)]
        self.assertEqual(len(clean_recent_history(rows,NOW)[0]),1)

    def test_order_and_outcome_or_odds_do_not_change_features(self):
        rows=[event(str(i),age=10+i) for i in range(10)]
        old=copy.deepcopy(rows)
        a,_=fresh_sequence_features(rows,'home',NOW,team_id='h')
        changed=list(reversed(copy.deepcopy(rows)))
        for row in changed:
            row.update(actual=2,odd=99,name='winner',rest_days=30)
        b,_=fresh_sequence_features(changed,'home',NOW,team_id='h')
        self.assertEqual(a,b)
        self.assertEqual(rows,old)


if __name__=='__main__':
    unittest.main()
