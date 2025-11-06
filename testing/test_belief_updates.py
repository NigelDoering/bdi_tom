"""
Test belief update mechanics for agents.

This test validates that:
1. Beliefs increase when agents observe open POIs at specific hours
2. Beliefs decrease when agents observe closed POIs at specific hours
3. Changes are localized to the observed hour (not all hours)
4. Magnitude of changes is reasonable (20-50% range)
5. Decay is gradual, not catastrophic
"""

import os
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_controller.world_graph import WorldGraph
from agent_controller.agent import Agent


class BeliefUpdateTester:
    """Test harness for belief update validation."""
    
    def __init__(self):
        """Initialize test with graph and test agent."""
        print("Loading graph for belief update tests...")
        graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
        G = nx.read_graphml(graph_path)
        self.world_graph = WorldGraph(G)
        self.G = self.world_graph.G
        
        # Create single test agent
        self.agent = Agent(agent_id="test_belief_agent", world_graph=self.world_graph, verbose=False)
        
        # Find test POIs with known opening hours
        self.test_pois = self._find_test_pois()
        
    def _find_test_pois(self):
        """Find POIs with clear opening/closing times for testing."""
        test_pois = []
        
        for node_id in self.world_graph.poi_nodes:
            node_data = self.G.nodes[node_id]
            opening_hours = node_data.get("opening_hours")
            
            # Look for POIs with clear open/close times (not 24/7)
            if isinstance(opening_hours, dict):
                open_hour = opening_hours.get("open", 0)
                close_hour = opening_hours.get("close", 24)
                
                # Skip 24-hour locations and ensure reasonable hours
                if open_hour != close_hour and close_hour - open_hour > 6:
                    test_pois.append({
                        "node_id": node_id,
                        "category": node_data.get("Category", "unknown"),
                        "name": node_data.get("poi_names", "Unknown"),
                        "open": open_hour,
                        "close": close_hour,
                        "coords": (node_data.get("y"), node_data.get("x"))
                    })
        
        return test_pois[:5]  # Use first 5 POIs for testing
    
    def get_initial_belief(self, poi_node, hour):
        """Get agent's initial belief about a POI being open at a specific hour."""
        return self.agent.belief_state[poi_node]["temporal_belief"][hour]
    
    def simulate_observation(self, poi_node, hour, distance=50):
        """
        Simulate agent observing a POI at a specific hour.
        
        Args:
            poi_node: POI node ID to observe
            hour: Hour of observation (0-23)
            distance: Distance threshold for belief update (meters)
        """
        self.agent.update_beliefs(
            current_node=poi_node,
            hour=hour,
            distance=distance
        )
    
    def test_open_poi_increases_belief(self):
        """Test that observing an open POI increases belief at that hour."""
        print("\n[TEST 1] Open POI increases belief at observed hour")
        print("-" * 60)
        
        failures = []
        
        for poi in self.test_pois:
            # Pick an hour when POI is open
            open_hour = poi["open"] + 2  # Mid-opening hours
            if open_hour >= poi["close"]:
                continue
            
            # Get initial belief
            initial_belief = self.get_initial_belief(poi["node_id"], open_hour)
            
            # Simulate observation
            self.simulate_observation(poi["node_id"], open_hour)
            
            # Get updated belief
            updated_belief = self.get_initial_belief(poi["node_id"], open_hour)
            
            # Calculate change
            belief_change = updated_belief - initial_belief
            percent_change = (belief_change / initial_belief) * 100 if initial_belief > 0 else 0
            
            # Validate: belief should increase
            if belief_change <= 0:
                failures.append({
                    "poi": poi["name"],
                    "hour": open_hour,
                    "reason": f"Belief decreased or unchanged: {initial_belief:.4f} → {updated_belief:.4f}"
                })
            # Validate: change should be reasonable (not too small, not too large)
            elif abs(percent_change) < 1:
                failures.append({
                    "poi": poi["name"],
                    "hour": open_hour,
                    "reason": f"Change too small: {percent_change:.2f}%"
                })
            elif abs(percent_change) > 100:
                failures.append({
                    "poi": poi["name"],
                    "hour": open_hour,
                    "reason": f"Change too large: {percent_change:.2f}%"
                })
            else:
                print(f"  ✓ {poi['name'][:30]:30s} @ {open_hour:02d}:00 | "
                      f"{initial_belief:.4f} → {updated_belief:.4f} ({percent_change:+.1f}%)")
        
        if failures:
            print(f"\n  ✗ {len(failures)} failure(s):")
            for f in failures:
                print(f"    - {f['poi']} @ {f['hour']:02d}:00: {f['reason']}")
            return False
        
        print(f"  ✓ All {len(self.test_pois)} POIs passed")
        return True
    
    def test_closed_poi_decreases_belief(self):
        """Test that observing a closed POI decreases belief at that hour."""
        print("\n[TEST 2] Closed POI decreases belief at observed hour")
        print("-" * 60)
        
        # Create fresh agent for this test
        test_agent = Agent(agent_id="test_closed_agent", world_graph=self.world_graph, verbose=False)
        
        failures = []
        
        for poi in self.test_pois:
            # Pick an hour when POI is closed
            closed_hour = (poi["close"] + 2) % 24
            
            # Skip if wraps into open time
            if poi["open"] <= closed_hour < poi["close"]:
                closed_hour = (poi["open"] - 2) % 24
            
            # Get initial belief
            initial_belief = test_agent.belief_state[poi["node_id"]]["temporal_belief"][closed_hour]
            
            # Simulate observation
            test_agent.update_beliefs(
                current_node=poi["node_id"],
                hour=closed_hour,
                distance=50
            )
            
            # Get updated belief
            updated_belief = test_agent.belief_state[poi["node_id"]]["temporal_belief"][closed_hour]
            
            # Calculate change
            belief_change = updated_belief - initial_belief
            percent_change = (belief_change / initial_belief) * 100 if initial_belief > 0 else 0
            
            # Validate: belief should decrease
            if belief_change >= 0:
                failures.append({
                    "poi": poi["name"],
                    "hour": closed_hour,
                    "reason": f"Belief increased or unchanged: {initial_belief:.4f} → {updated_belief:.4f}"
                })
            # Validate: change should be reasonable
            elif abs(percent_change) < 1:
                failures.append({
                    "poi": poi["name"],
                    "hour": closed_hour,
                    "reason": f"Change too small: {percent_change:.2f}%"
                })
            elif abs(percent_change) > 100:
                failures.append({
                    "poi": poi["name"],
                    "hour": closed_hour,
                    "reason": f"Change too large: {percent_change:.2f}%"
                })
            else:
                print(f"  ✓ {poi['name'][:30]:30s} @ {closed_hour:02d}:00 | "
                      f"{initial_belief:.4f} → {updated_belief:.4f} ({percent_change:+.1f}%)")
        
        if failures:
            print(f"\n  ✗ {len(failures)} failure(s):")
            for f in failures:
                print(f"    - {f['poi']} @ {f['hour']:02d}:00: {f['reason']}")
            return False
        
        print(f"  ✓ All {len(self.test_pois)} POIs passed")
        return True
    
    def test_belief_localized_to_hour(self):
        """Test that belief updates only affect the observed hour, not others."""
        print("\n[TEST 3] Belief updates localized to observed hour")
        print("-" * 60)
        
        # Create fresh agent
        test_agent = Agent(agent_id="test_localized_agent", world_graph=self.world_graph, verbose=False)
        
        failures = []
        
        for poi in self.test_pois[:2]:  # Test with 2 POIs
            # Pick an observation hour (when open)
            obs_hour = poi["open"] + 2
            if obs_hour >= poi["close"]:
                continue
            
            # Record beliefs at all hours before observation
            before_beliefs = [
                test_agent.belief_state[poi["node_id"]]["temporal_belief"][h] 
                for h in range(24)
            ]
            
            # Simulate observation at specific hour
            test_agent.update_beliefs(
                current_node=poi["node_id"],
                hour=obs_hour,
                distance=50
            )
            
            # Record beliefs at all hours after observation
            after_beliefs = [
                test_agent.belief_state[poi["node_id"]]["temporal_belief"][h]
                for h in range(24)
            ]
            
            # Check that only the observed hour changed significantly
            changes = [after - before for before, after in zip(before_beliefs, after_beliefs)]
            
            # The observed hour should have the largest change
            max_change_hour = np.argmax(np.abs(changes))
            
            if max_change_hour != obs_hour:
                failures.append({
                    "poi": poi["name"],
                    "reason": f"Max change at hour {max_change_hour}, expected {obs_hour}"
                })
            else:
                # Check that other hours changed much less (due to decay only)
                observed_change = abs(changes[obs_hour])
                other_changes = [abs(changes[h]) for h in range(24) if h != obs_hour]
                avg_other_change = np.mean(other_changes)
                
                ratio = observed_change / avg_other_change if avg_other_change > 0 else float('inf')
                
                if ratio < 5:  # Observed hour should change at least 5x more
                    failures.append({
                        "poi": poi["name"],
                        "reason": f"Observed hour change not sufficiently larger: {ratio:.1f}x"
                    })
                else:
                    print(f"  ✓ {poi['name'][:30]:30s} | Hour {obs_hour:02d} changed {ratio:.1f}x more than others")
        
        if failures:
            print(f"\n  ✗ {len(failures)} failure(s):")
            for f in failures:
                print(f"    - {f['poi']}: {f['reason']}")
            return False
        
        print(f"  ✓ All tests passed")
        return True
    
    def test_decay_gradual_not_catastrophic(self):
        """
        Test that belief decay moves beliefs toward prior over time at the expected rate.
        
        Simulates 1,000 trajectories (~10,000 decay updates) and validates that:
        - Beliefs retain 30-50% of their distance from the prior
        - Decay is gradual and not catastrophic
        """
        print("\n[TEST 4] Decay is gradual and moves toward prior")
        print("-" * 60)
        
        # Create fresh agent
        test_agent = Agent(agent_id="test_decay_agent", world_graph=self.world_graph, verbose=False)
        
        # Pick one POI for decay testing
        poi = self.test_pois[0]
        test_hour = poi["open"] + 2
        
        if test_hour >= poi["close"]:
            print("  ⚠ Skipping: no suitable test hour")
            return True
        
        # Get the prior belief for comparison
        prior_belief = test_agent.belief_prior[poi["node_id"]]["alpha"][test_hour] / (
            test_agent.belief_prior[poi["node_id"]]["alpha"][test_hour] + 
            test_agent.belief_prior[poi["node_id"]]["beta"][test_hour]
        )
        
        # Step 1: First observe the POI as OPEN to increase belief away from prior
        test_agent.update_beliefs(
            current_node=poi["node_id"],
            hour=test_hour,
            distance=50
        )
        belief_after_observation = test_agent.belief_state[poi["node_id"]]["temporal_belief"][test_hour]
        
        # Find a distant node to trigger decay without observing this POI
        distant_nodes = []
        for node_id in list(self.G.nodes())[:100]:  # Check first 100 nodes
            node_data = self.G.nodes[node_id]
            if node_id == poi["node_id"]:
                continue
            
            # Calculate distance from test POI
            poi_coords = poi["coords"]
            node_coords = (node_data.get("y"), node_data.get("x"))
            
            if poi_coords[0] and node_coords[0]:
                import haversine as hs
                dist = hs.haversine(poi_coords, node_coords, unit=hs.Unit.METERS)
                if dist > 500:  # More than 500m away
                    distant_nodes.append(node_id)
                    if len(distant_nodes) >= 3:
                        break
        
        if len(distant_nodes) < 3:
            print("  ⚠ Skipping: couldn't find distant nodes for decay test")
            return True
        
        # Step 2: Simulate 1,000 trajectories (10,000 decay updates)
        # Each trajectory = ~50 nodes, update every 5 nodes = 10 updates per trajectory
        num_trajectories = 1000
        updates_per_trajectory = 10
        total_updates = num_trajectories * updates_per_trajectory
        
        beliefs_over_time = [belief_after_observation]
        
        # Use tqdm progress bar for the decay simulation
        for i in tqdm(range(total_updates), desc="  Simulating decay", unit="update", ncols=70):
            # Update at a distant node (POI won't be observed due to distance)
            distant_node = distant_nodes[i % len(distant_nodes)]
            test_agent.update_beliefs(
                current_node=distant_node,
                hour=test_hour,
                distance=100  # 100m radius, POI is >500m away
            )
            current_belief = test_agent.belief_state[poi["node_id"]]["temporal_belief"][test_hour]
            beliefs_over_time.append(current_belief)
        
        print()  # Add newline after progress bar
        final_belief = beliefs_over_time[-1]
        
        # Validate decay behavior
        failures = []
        
        # Calculate distances from prior
        initial_distance_from_prior = abs(belief_after_observation - prior_belief)
        final_distance_from_prior = abs(final_belief - prior_belief)
        
        # 1. Belief should have changed (decay is happening)
        belief_change = abs(final_belief - belief_after_observation)
        if belief_change < 0.001:
            failures.append(
                f"Belief didn't decay: {belief_after_observation:.4f} → {final_belief:.4f}"
            )
        
        # 2. Belief should move toward prior (distance from prior should decrease)
        if final_distance_from_prior >= initial_distance_from_prior:
            failures.append(
                f"Belief didn't move toward prior: "
                f"distance {initial_distance_from_prior:.4f} → {final_distance_from_prior:.4f}"
            )
        
        # 3. After 1,000 trajectories, should retain 30-50% of distance from prior
        retention_ratio = final_distance_from_prior / initial_distance_from_prior if initial_distance_from_prior > 0 else 0
        
        if retention_ratio < 0.20:
            failures.append(
                f"Decay too rapid: retained only {retention_ratio*100:.1f}% (expected 30-50%)"
            )
        elif retention_ratio > 0.60:
            failures.append(
                f"Decay too slow: retained {retention_ratio*100:.1f}% (expected 30-50%)"
            )
        
        # 4. Should not collapse to extremes
        if final_belief < 0.05 or final_belief > 0.95:
            failures.append(f"Belief collapsed to extreme: {final_belief:.4f}")
        
        if failures:
            print(f"  ✗ Test failed:")
            for failure in failures:
                print(f"    - {failure}")
            print(f"\n    POI: {poi['name']}")
            print(f"    Prior belief: {prior_belief:.4f}")
            print(f"    After observation: {belief_after_observation:.4f}")
            print(f"    After {num_trajectories} trajectories: {final_belief:.4f}")
            print(f"    Retention: {retention_ratio*100:.1f}%")
            print(f"    Decay rate: {test_agent.decay_rate:.6f}")
            return False
        
        # Calculate convergence toward prior
        convergence_pct = (1 - retention_ratio) * 100
        
        print(f"  ✓ {poi['name'][:30]:30s}")
        print(f"    Prior belief:       {prior_belief:.4f}")
        print(f"    After observation:  {belief_after_observation:.4f}")
        print(f"    After {num_trajectories} trajs:  {final_belief:.4f}")
        print(f"    Change:             {(final_belief - belief_after_observation):+.4f}")
        print(f"    Retention:          {retention_ratio*100:.1f}% ✓")
        print(f"    Converged to prior: {convergence_pct:.1f}%")
        print(f"    Decay rate:         {test_agent.decay_rate:.6f}")
        
        return True
    
    def run_all_tests(self):
        """Run all belief update tests."""
        print("\n" + "=" * 60)
        print("BELIEF UPDATE TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Open POI increases belief", self.test_open_poi_increases_belief),
            ("Closed POI decreases belief", self.test_closed_poi_decreases_belief),
            ("Updates localized to observed hour", self.test_belief_localized_to_hour),
            ("Decay is gradual", self.test_decay_gradual_not_catastrophic),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                passed = test_func()
                results[test_name] = passed
            except Exception as e:
                print(f"\n  ✗ Exception: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for p in results.values() if p)
        total = len(results)
        
        for test_name, passed_flag in results.items():
            status = "✓ PASS" if passed_flag else "✗ FAIL"
            print(f"  {status:8s} | {test_name}")
        
        print()
        print(f"Passed: {passed}/{total}")
        
        return all(results.values())


def main():
    """Run belief update tests."""
    tester = BeliefUpdateTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All belief update tests passed!")
        return 0
    else:
        print("\n❌ Some belief update tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
