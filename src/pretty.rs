use std::fmt::Debug;

use crate::algo::Schedule;

pub fn pretty(schedule: Schedule) -> String {
    format!("{:#?}", schedule)
}
