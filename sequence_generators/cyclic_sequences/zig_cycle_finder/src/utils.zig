//! Author: Hearues Zueke
//! E-Mail: hziko314@gmail.com
//! Datetime: 2026.01.22T19:12+1

const std = @import("std");

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer: std.fs.File.Writer = std.fs.File.stdout().writer(&stdout_buffer);
pub const stdout: *std.Io.Writer = &stdout_writer.interface;

pub fn printU8ArrayAsHex(arr: []const u8) !void {
    try stdout.print("{X:0>2}", .{arr[0]});
    for (arr[1..]) |value| {
        try stdout.print(",{X:0>2}", .{value});
    }
}

pub fn printU8ArrayAsDec(arr: []const u8) !void {
    try stdout.print("{d}", .{arr[0]});
    for (arr[1..]) |value| {
        try stdout.print(",{d}", .{value});
    }
}

pub fn printU32ArrayAsHex(arr: []const u32) !void {
    try stdout.print("{X:0>8}", .{arr[0]});
    for (arr[1..]) |value| {
        try stdout.print(",{X:0>8}", .{value});
    }
}
