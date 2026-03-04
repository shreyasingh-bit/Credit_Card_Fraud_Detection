import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { motion } from "framer-motion";

export default function FraudDetectionApp() {
  const [amount, setAmount] = useState("");
  const [location, setLocation] = useState("");
  const [device, setDevice] = useState("");
  const [result, setResult] = useState(null);

  const analyzeTransaction = () => {
    // Simple rule-based mock model
    let riskScore = 0;
    if (amount > 1000) riskScore += 40;
    if (location === "International") riskScore += 30;
    if (device === "Unknown") riskScore += 30;

    setResult({
      fraud: riskScore >= 60,
      score: riskScore,
    });
  };

  const barData = result
    ? [
        { name: "Risk Score", value: result.score },
        { name: "Safe Score", value: 100 - result.score },
      ]
    : [];

  const pieData = result
    ? [
        { name: "Fraud Risk", value: result.score },
        { name: "Remaining", value: 100 - result.score },
      ]
    : [];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl font-bold mb-6 text-center"
      >
        Credit Card Fraud Detection System
      </motion.h1>

      <Card className="max-w-xl mx-auto mb-6">
        <CardContent className="space-y-4 p-6">
          <Input
            placeholder="Transaction Amount"
            type="number"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
          <Input
            placeholder="Location (Local / International)"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
          <Input
            placeholder="Device (Known / Unknown)"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
          />
          <Button className="w-full" onClick={analyzeTransaction}>
            Detect Fraud
          </Button>
        </CardContent>
      </Card>

      {result && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          <Card>
            <CardContent className="p-4">
              <h2 className="text-xl font-semibold mb-2">Detection Result</h2>
              <p className="text-lg">
                Status: {result.fraud ? "⚠️ Fraud Detected" : "✅ Legitimate Transaction"}
              </p>
              <p>Risk Score: {result.score}%</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4 h-64">
              <h2 className="text-xl font-semibold mb-2">Risk Analysis (Bar)</h2>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardContent className="p-4 h-64">
              <h2 className="text-xl font-semibold mb-2">Fraud Probability (Pie)</h2>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" outerRadius={80} label>
                    {pieData.map((_, index) => (
                      <Cell key={index} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
