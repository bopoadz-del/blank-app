"""
Database migration: Add credibility_tier field to formulas table.

Run this to add the tier field to existing database.
"""
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, Formula
from app.core.config import settings

def add_credibility_tier_column():
    """Add credibility_tier column to formulas table."""
    
    engine = create_engine(str(settings.DATABASE_URL))
    
    # Add column using raw SQL (for existing databases)
    with engine.connect() as conn:
        try:
            # Check if column exists
            result = conn.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='formulas' AND column_name='credibility_tier';
            """)
            
            if not result.fetchone():
                # Add column
                conn.execute("""
                    ALTER TABLE formulas 
                    ADD COLUMN credibility_tier VARCHAR(50) DEFAULT 'tier_5_sandbox_only';
                """)
                
                conn.execute("""
                    ALTER TABLE formulas 
                    ADD COLUMN tier_updated_at TIMESTAMP;
                """)
                
                conn.execute("""
                    ALTER TABLE formulas 
                    ADD COLUMN tier_change_reason TEXT;
                """)
                
                conn.commit()
                print("✅ Added credibility_tier fields to formulas table")
            else:
                print("✅ credibility_tier column already exists")
        
        except Exception as e:
            print(f"❌ Error adding column: {e}")
            conn.rollback()

def initialize_tiers_for_existing_formulas():
    """Set initial tiers for existing formulas based on source."""
    from app.services.credibility_tiers import credibility_manager
    
    engine = create_engine(str(settings.DATABASE_URL))
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        formulas = db.query(Formula).all()
        
        for formula in formulas:
            # Determine initial tier
            tier = credibility_manager.determine_initial_tier(
                source=formula.source or "user_submitted",
                validation_passed=len(formula.validation_stages_passed or []) >= 4
            )
            
            formula.credibility_tier = tier.value
            formula.tier_updated_at = datetime.utcnow()
            formula.tier_change_reason = "Initial tier assignment based on source"
        
        db.commit()
        print(f"✅ Initialized tiers for {len(formulas)} formulas")
    
    except Exception as e:
        print(f"❌ Error initializing tiers: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    from datetime import datetime
    
    print("=== Database Migration: Add Credibility Tiers ===\n")
    
    # Step 1: Add column
    add_credibility_tier_column()
    
    # Step 2: Initialize tiers for existing formulas
    initialize_tiers_for_existing_formulas()
    
    print("\n=== Migration Complete ===")
