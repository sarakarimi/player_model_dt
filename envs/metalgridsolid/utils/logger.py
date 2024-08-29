class Logger:
    def __init__(self):
        self.steps_taken = 0
        self.enemies_killed = 0
        self.items_used = []
        self.items_dropped = []

    def log_step(self):
        """Log a step taken by the agent."""
        self.steps_taken += 1

    def log_enemy_killed(self):
        """Log an enemy being killed."""
        self.enemies_killed += 1

    def log_item_used(self, item_name):
        """Log an item being used by the agent."""
        self.items_used.append(item_name)
        
    def log_item_dropped(self, item_name):
        """Log an item being dropped by the agent."""
        self.items_dropped.append(item_name)

    def reset(self):
        """Reset all logs for a new game/episode."""
        self.steps_taken = 0
        self.enemies_killed = 0
        self.items_used.clear()
        self.items_dropped.clear()

    def get_summary(self):
        """Return a summary of the logged events."""
        return {
            "steps_taken": self.steps_taken,
            "enemies_killed": self.enemies_killed,
            "items_used": self.items_used,
            "items_dropped": self.items_dropped
        }

    def print_summary(self):
        """Print the summary of the logged events."""
        summary = self.get_summary()
        print("Game Summary:")
        print(f"Steps Taken: {summary['steps_taken']}")
        print(f"Enemies Killed: {summary['enemies_killed']}")
        print(f"Items Used: {summary['items_used']}")
        print(f"Items Dropped: {summary['items_dropped']}")